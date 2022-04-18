# ECSE 552 HW4

Repository containing code for HW 4.

To setup in colab:

```python
# Download and setup repo.

%%capture
!pip install pytorch_lightning
!git clone https://github.com/maxsolomonhenry/ecse-552-hw4.git
import sys
sys.path.append('/content/ecse-552-hw4/src/')
sys.path.append('/content/ecse-552-hw4/data/')

# Download data (only online for the duration of this assignment).
!curl https://transfer.sh/11hWAU/weather_train.csv -o /content/ecse-552-hw4/data/weather_train.csv

# Change working directory to repo directory.
%cd /content/ecse-552-hw4

``` 

Example code use:
```python
from data import get_dataloaders
from model import BaselineMlp

train_loader, val_loader = get_dataloaders(
    n_past=8, batch_size=128, percent_train=0.8
)

train_iter = iter(train_loader)
x, y = next(train_iter)

n_input = x.shape[1]
n_output = y.shape[1]

n_hidden = [256, 128, 64, 32]

model = BaselineMlp(
    n_input=n_input, n_hidden=n_hidden, n_output=n_output
)

print(model)
```
