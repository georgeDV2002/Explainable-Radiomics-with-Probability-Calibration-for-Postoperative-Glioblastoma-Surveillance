#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

fontsize = 10

# Confusion matrix values
cm = np.array([[27, 16],
               [6, 47]])

labels = ["Class 0", "Class 1"]

fig, ax = plt.subplots(figsize=(3, 3))
im = ax.imshow(cm, cmap="Blues")

# Show numbers in cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="black", fontsize=fontsize)

# Ticks and labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels([f"Pred {l}" for l in labels], fontsize=fontsize - 3)
ax.set_yticklabels([f"True {l}" for l in labels], fontsize=fontsize - 3)

# Remove grid lines & make elegant
ax.spines[:].set_visible(False)
plt.tight_layout()

# Save as high-quality PDF/PNG for papers
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
 
