from bs4 import BeautifulSoup
import scrapy



class covid_crawler(scrapy.Spider):
    name = "cs6440_covid_bot"
    start_urls = [
        'https://www.who.int/news-room/q-a-detail',
    ]

    def parse(self, response):
        for quote in response.css('div.sf-accordion__panel'):
            yield {
                'question': quote.css('a.sf-accordion__link::text').get().strip(),
                'answer': BeautifulSoup(quote.css('div.sf-accordion__content').get(), "lxml").text.strip()
            }

        for next_page in response.css('a.sf-list-vertical__item'):
            print("move to next page : {}".format(next_page))
            yield response.follow(next_page, self.parse)