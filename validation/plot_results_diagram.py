# Plot how often intent labels appear on their own and in combination.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

intent_counts = pd.read_csv("data/intent_classification_aggregated.csv", index_col="intents", keep_default_na=False)["count"]
label_names = {
    "fix_vulnerability": "Fix",
    "find_vulnerabilities": "Find",
    "prevent_vulnerabilities": "Prevent",
    "explain_vulnerability": "Explain",
    "verify_vulnerability": "Verify",
    "other_security": "Other",
}
assert intent_counts.sum() == 5825

membership = pd.DataFrame(
    [{label: label in labels.split("; ") for label in label_names} for labels in intent_counts.index]
).assign(count=intent_counts.to_numpy())
membership = (
    membership[membership[list(label_names)].any(axis=1)]
    .groupby(list(label_names), as_index=False, sort=False)["count"]
    .sum()
    .sort_values("count", ascending=False, kind="stable")
)

label_totals = membership[list(label_names)].mul(membership["count"], axis=0).sum().sort_values(ascending=False)
labels = list(label_totals.index)
has_fix = membership["fix_vulnerability"].to_numpy()

light, dark, absent = "#b7a9fa", "#2a224b", "#ececec"
column_colors = np.where(has_fix, dark, light)
columns = np.arange(len(membership))
rows = np.arange(len(labels))[::-1]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10 / 2, 4.8 / 2),
    gridspec_kw=dict(width_ratios=[1, 4.5], height_ratios=[2, 1.4], wspace=0.45, hspace=0.05),
)
axes[0, 0].axis("off")
ax_bars, ax_totals, ax_matrix = axes[0, 1], axes[1, 0], axes[1, 1]

ax_bars.bar(columns, membership["count"], color=column_colors, width=0.7)
for x, count in zip(columns, membership["count"]):
    ax_bars.text(x, count + 60, f"{count:,}", ha="center", va="bottom", fontsize=5.5, rotation=90)
ax_bars.set_ylim(0, 3_300)
ax_bars.set_ylabel("Prompts with\nlabel combination", fontsize=8)
ax_bars.set_xlim(-0.6, len(membership) - 0.4)
ax_bars.set_xticks([])
ax_bars.spines[["top", "right", "bottom"]].set_visible(False)
ax_bars.tick_params(labelsize=7)
ax_bars.legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=dark), plt.Rectangle((0, 0), 1, 1, color=light)],
    labels=["Includes Fix", "Without Fix"],
    frameon=False,
    fontsize=8,
    loc="upper right",
)

for y, label in zip(rows, labels):
    if y % 2 == 0:
        ax_matrix.axhspan(y - 0.5, y + 0.5, color="#f5f5f5", zorder=0)
    present = membership[label].to_numpy()
    ax_matrix.scatter(columns[~present], np.full((~present).sum(), y), s=12, color=absent, zorder=1)
    ax_matrix.scatter(columns[present], np.full(present.sum(), y), s=12, color=column_colors[present], zorder=3)
label_rows = dict(zip(labels, rows))
for x, (_, row) in zip(columns, membership.iterrows()):
    present_rows = [label_rows[label] for label in labels if row[label]]
    ax_matrix.plot([x, x], [min(present_rows), max(present_rows)], color=column_colors[x], lw=1.5, zorder=2)
ax_matrix.set_xlim(ax_bars.get_xlim())
ax_matrix.set_ylim(-0.5, len(labels) - 0.5)
ax_matrix.set_xticks([])
ax_matrix.set_yticks(rows, [label_names[label] for label in labels], fontsize=8)
for tick_label in ax_matrix.get_yticklabels():
    tick_label.set_color("#2a224b")
ax_matrix.get_yticklabels()[labels.index("fix_vulnerability")].set_fontweight("bold")
ax_matrix.tick_params(axis="y", length=0, pad=6)
ax_matrix.spines[:].set_visible(False)

ax_totals.barh(rows, label_totals, color=[dark if label == "fix_vulnerability" else light for label in labels], height=0.6)
for y, total in zip(rows, label_totals):
    ax_totals.text(total + 80, y, f"{total:,}", ha="right", va="center", fontsize=7, transform=ax_totals.transData)
ax_totals.invert_xaxis()
ax_totals.set_ylim(ax_matrix.get_ylim())
ax_totals.set_yticks([])
ax_totals.tick_params(axis="x", labelsize=7)
ax_totals.set_xlabel("Prompts with label", fontsize=8)
ax_totals.spines[["top", "left", "right"]].set_visible(False)

fig.savefig("data/intent_upset.pdf", bbox_inches="tight")
fig.savefig("data/intent_upset.png", bbox_inches="tight", dpi=200)
plt.show()
