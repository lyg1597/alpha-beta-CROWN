import torch
import torch.nn as nn

# Two toy models:
model1 = nn.Linear(2, 1)
model2 = nn.Linear(2, 1)

x1 = torch.randn(2, requires_grad=True)
x2 = torch.randn(2, requires_grad=True)

# Forward + backward pass for model1:
loss1 = model1(x1).sum()  # Just some toy loss
loss1.backward(retain_graph=False)  # Freed the graph
# If we do NOT retain the graph, we can't backprop through loss1 again.

# Now we want to use `loss1` in the next loss, but treat it as a constant:
loss2 = model2(x1+x2).sum() + loss1.detach()  # detach() => treat loss1 as constant
loss2.backward()  # This works fine, no "second backward" error