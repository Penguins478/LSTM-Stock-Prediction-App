from django.db import models

class Stock(models.Model):
    
    ticker_symbol = models.CharField(max_length=10)
    company_name = models.CharField(max_length=100)

    def __str__(self):
        return f"{self.ticker_symbol} - {self.company_name}"
