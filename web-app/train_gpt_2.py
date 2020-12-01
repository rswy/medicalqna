# CREDIT GOES TO THIS notebook which I used as reference to test train my model on : https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=x0WeP5PREUuy


from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import datetime


batch_size = 2
  # Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 


def prepare_training_data(data):
  # Prepare the training / Validation Data
  
  data_answers = data.answer.copy().tolist()
  data_questions = data.question.copy().tolist()
  data_qna = data_answers+data_questions

  dataset = GPT2Dataset(data_qna, tokenizer, max_length=768)

  # Split into training and validation sets
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  print('{:>5,} training samples'.format(train_size))
  print('{:>5,} validation samples'.format(val_size))


  # Create the DataLoaders for our training and validation datasets.
  # We'll take training samples in random order. 
  train_dataloader = DataLoader(
              train_dataset,  # The training samples.
              sampler = RandomSampler(train_dataset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )

  # For validation the order doesn't matter, so we'll just read them sequentially.
  validation_dataloader = DataLoader(
              val_dataset, # The validation samples.
              sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
              batch_size = batch_size # Evaluate with this batch size.
          )


def fine_tune_gpt2():
  # I'm not really doing anything with the config buheret
  configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

  # instantiate the model
  model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

  # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
  # otherwise the tokenizer and model tensors won't match up
  model.resize_token_embeddings(len(tokenizer))

  # Tell pytorch to run this model on the GPU.
  device = torch.device("cuda")
  model.cuda()

  # Set the seed value all over the place to make this reproducible.
  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)



  # some parameters I cooked up that work reasonably well

  epochs = 5
  learning_rate = 5e-4
  warmup_steps = 1e2
  epsilon = 1e-8

  # this produces sample output every 100 steps
  sample_every = 100

  # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
  optimizer = AdamW(model.parameters(),
                    lr = learning_rate,
                    eps = epsilon
                  )

  # Total number of training steps is [number of batches] x [number of epochs]. 
  # (Note that this is not the same as the number of training samples).
  total_steps = len(train_dataloader) * epochs

  # Create the learning rate scheduler.
  # This changes the learning rate as the training loop progresses
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = warmup_steps, 
                                              num_training_steps = total_steps)


  def format_time(elapsed):
      return str(datetime.timedelta(seconds=int(round((elapsed)))))



  total_t0 = time.time()

  training_stats = []

  model = model.to(device)

  for epoch_i in range(0, epochs):

      # ========================================
      #               Training
      # ========================================

      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      t0 = time.time()

      total_train_loss = 0

      model.train()

      for step, batch in enumerate(train_dataloader):

          b_input_ids = batch[0].to(device)
          b_labels = batch[0].to(device)
          b_masks = batch[1].to(device)

          model.zero_grad()        

          outputs = model(  b_input_ids,
                            labels=b_labels, 
                            attention_mask = b_masks,
                            token_type_ids=None
                          )

          loss = outputs[0]  

          batch_loss = loss.item()
          total_train_loss += batch_loss

          # Get sample every x batches.
          if step % sample_every == 0 and not step == 0:

              elapsed = format_time(time.time() - t0)
              print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

              model.eval()

              sample_outputs = model.generate(
                                      bos_token_id=random.randint(1,30000),
                                      do_sample=True,   
                                      top_k=50, 
                                      max_length = 200,
                                      top_p=0.95, 
                                      num_return_sequences=1
                                  )
              for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
              
              model.train()

          loss.backward()

          optimizer.step()

          scheduler.step()

      # Calculate the average loss over all of the batches.
      avg_train_loss = total_train_loss / len(train_dataloader)       
      
      # Measure how long this epoch took.
      training_time = format_time(time.time() - t0)

      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Training epoch took: {:}".format(training_time))
          
      # ========================================
      #               Validation
      # ========================================

      print("")
      print("Running Validation...")

      t0 = time.time()

      model.eval()

      total_eval_loss = 0
      nb_eval_steps = 0

      # Evaluate data for one epoch
      for batch in validation_dataloader:
          
          b_input_ids = batch[0].to(device)
          b_labels = batch[0].to(device)
          b_masks = batch[1].to(device)
          
          with torch.no_grad():        

              outputs  = model(b_input_ids, 
  #                            token_type_ids=None, 
                              attention_mask = b_masks,
                              labels=b_labels)
            
              loss = outputs[0]  
              
          batch_loss = loss.item()
          total_eval_loss += batch_loss        

      avg_val_loss = total_eval_loss / len(validation_dataloader)
      
      validation_time = format_time(time.time() - t0)    

      print("  Validation Loss: {0:.2f}".format(avg_val_loss))
      print("  Validation took: {:}".format(validation_time))

      # Record all statistics from this epoch.
      training_stats.append(
          {
              'epoch': epoch_i + 1,
              'Training Loss': avg_train_loss,
              'Valid. Loss': avg_val_loss,
              'Training Time': training_time,
              'Validation Time': validation_time
          }
      )

  print("")
  print("Training complete!")
  print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



def save_and_load_model():
  # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

  output_dir = './model_save/'

  # Create output directory if needed
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  print("Saving model to %s" % output_dir)

  # Save a trained model, configuration and tokenizer using `save_pretrained()`.
  # They can then be reloaded using `from_pretrained()`
  model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
  model_to_save.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

  # Good practice: save your training arguments together with the trained model
  # torch.save(args, os.path.join(output_dir, 'training_args.bin'))


  !ls -l --block-size=K ./model_save/

  !ls -l --block-size=M ./model_save/pytorch_model.bin


  # Copy the model files to a directory in your Google Drive.
  !cp -r ./model_save/ $data_dir

  # Load a trained model and vocabulary that you have fine-tuned
  model = GPT2LMHeadModel.from_pretrained(output_dir)
  tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
  model.to(device)

  return model,tokenizer



def generate_text(model,tokenizer,prompt = "<|startoftext|>"):
  model.eval() 
  generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
  generated = generated.to(device)

  print(generated)

  sample_outputs = model.generate(
                                  generated, 
                                  #bos_token_id=random.randint(1,30000),
                                  do_sample=True,   
                                  top_k=50, 
                                  max_length = 300,
                                  top_p=0.95, 
                                  num_return_sequences=3
                                  )

  for i, sample_output in enumerate(sample_outputs):
    print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

  returns sameple_outputs
