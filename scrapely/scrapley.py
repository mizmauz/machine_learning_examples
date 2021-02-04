from scrapely import Scraper
s = Scraper()

url1 = 'http://pypi.python.org/pypi/w3lib/1.1'
data = {'name': 'w3lib 1.1', 'author': 'Scrapy project', 'description': 'Library of web-related functions'}
s.train(url1, data)

url2 = 'http://pypi.python.org/pypi/Django/1.3'
s.scrape(url2)
[{u'author': [u'Django Software Foundation &lt;foundation at djangoproject com&gt;'],
  u'description': [u'A high-level Python Web framework that encourages rapid development and clean, pragmatic design.'],
  u'name': [u'Django 1.3']}]