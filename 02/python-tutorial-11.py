print("||-------------------------Python Modules------------------------------||")
import pricing
from pricing import get_net_price
import pricing as selling_price
from pricing import get_net_price as calculate_net_price

net_price = pricing.get_net_price(
    price=100,
    tax_rate=0.01
)

print(net_price)
print("||-----------------------------------------------------------------------||")
net_price = get_net_price(price=100, tax_rate=0.01)
print(net_price)
print("||-----------------------------------------------------------------------||")
net_price = calculate_net_price(
    price=100,
    tax_rate=0.1,
    discount=0.05
)
print("||-----------------------------------------------------------------------||")
from pricing import *
from product import *

tax = get_tax(100)
print(tax)

print("||-------------------------Python Module search path------------------------------||")
import sys

for path in sys.path:
    print(path)

import sys
sys.path.append('/Users/likhitha/Documents/ece5831-2024-assignments/ece5831-2024-assignments/02')

print("||-------------------------Python Name------------------------------||")

import billing

print("||-------------------------Python Packages------------------------------||")

import sales.order
import sales.delivery
import sales.billing


sales.order.create_sales_order()
sales.delivery.create_delivery()
sales.billing.create_billing()
print("||-----------------------------------------------------------------------||")

from sales.order import create_sales_order
from sales.delivery import create_delivery
from sales.billing import create_billing


create_sales_order()
create_delivery()
create_billing()
print("||-----------------------------------------------------------------------||")


from sales.order import create_sales_order as create_order
from sales.delivery import create_delivery as start_delivery
from sales.billing import create_billing as issue_billing


create_order()
start_delivery()
issue_billing()

print("||-----------------------------------------------------------------------||")

from sales import TAX_RATE

print(TAX_RATE)

print("||-----------------------------------------------------------------------||")


import sales

sales.order.create_sales_order()


print("||-----------------------------------------------------------------------||")

from sales import *


order.create_sales_order()
delivery.create_delivery()

print("||-----------------------------------------------------------------------||")

from sales.order import create_sales_order

create_sales_order()

