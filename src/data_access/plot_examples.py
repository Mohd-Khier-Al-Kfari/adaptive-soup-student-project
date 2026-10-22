"""Plot prepared UDA comparisons. Author: Moh'd Khier Al Kfari."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from load_data import choose_and_load

PREPARED = Path(__file__).resolve().parents[2] / "data" / "prepared"
ACTIVITIES = ("running", "walking", "standing")
LOCATIONS = {"PAMAP2": ("hand", "ankle"), "RealWorld": ("forearm", "shin"), "MHEALTH": ("forearm", "ankle")}
DISPLAY_LOCATIONS = ("wrist", "ankle")

def first_activity_window(item, activity):
    inverse={value:key for key,value in item["labels"]["label_map"].items()}
    for split, subjects in item["labels"]["labels"].items():
        for subject, labels in subjects.items():
            x=item["features"].get(split, {}).get(subject)
            if x is None: continue
            for index, value in enumerate(labels):
                if inverse[int(value)].lower() == activity: return np.asarray(x[index])
    return None

def activity_windows(item):
    """Index the first available window for each activity once per item."""
    return {activity: first_activity_window(item, activity) for activity in ACTIVITIES}

def item_identity(name, item):
    metadata = item.get("metadata") or {}
    dataset = metadata.get("dataset") or name.split("_", 1)[0]
    group = metadata.get("group")
    if group is None and "_" in name:
        group = name.split("_", 1)[1]
    return dataset, group

def plot_comparison(items, title, output_path, row_specs):
    row_count = len(row_specs)
    figure, axes=plt.subplots(
        row_count, 3,
        figsize=(15, max(8, 2.85 * row_count)),
        squeeze=False,
    )
    colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    cached = {name: activity_windows(item) for name, item in items.items()}
    for column, activity in enumerate(ACTIVITIES):
        for line_name, item in items.items():
            window=cached[line_name][activity]
            if window is None: continue
            dataset, group = item_identity(line_name, item)
            locations=LOCATIONS.get(dataset, ("upper", "lower"))
            for plot_row, spec in enumerate(row_specs):
                if spec["dataset"] != dataset or (spec.get("group") is not None and spec.get("group") != group):
                    continue
                offset = 0 if spec["location"] == DISPLAY_LOCATIONS[0] else 9
                for axis_index, axis_name in ((0, "x"), (1, "y"), (2, "z")):
                    color = colors[axis_name]
                    axes[plot_row,column].plot(window[:,offset+axis_index], color=color, linewidth=.85,
                                               alpha=.8, label=f"acc_{axis_name}")
        for plot_row, spec in enumerate(row_specs):
            axes[plot_row, column].set_title(
                " | ".join(spec["title_parts"] + [activity]), fontsize=15
            )
    for row in range(row_count):
        for cell in axes[row, :]:
            cell.set_xlabel("")
            cell.set_ylabel("")
            cell.grid(alpha=.25)
            cell.tick_params(axis="both", labelsize=11)
            legend = cell.get_legend()
            if legend is not None:
                legend.remove()
    handles = [Line2D([0], [0], color=colors[name], linewidth=1.2, label=f"acc_{name}")
               for name in ("x", "y", "z")]
    figure.suptitle(title, fontsize=19, y=0.995)
    figure.supxlabel("Sample", fontsize=16, y=0.018)
    figure.supylabel("Acceleration", fontsize=16, x=0.018)
    figure.legend(handles=handles, labels=["acc_x", "acc_y", "acc_z"],
                  loc="upper center", ncol=3, frameon=True, fontsize=13,
                  bbox_to_anchor=(0.5, 0.965))
    figure.tight_layout(rect=(0.045, 0.04, 0.995, 0.925), h_pad=0.08, w_pad=0.08)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output_path,dpi=150)
    plt.show()
    plt.close(figure)
    return output_path

def plot_selected_experiment(data, experiment):
    output=PREPARED/"plots"; paths=[]
    if experiment == "cross_dataset":
        order = ("PAMAP2", "RealWorld", "MHEALTH")
        rows = [{"dataset": dataset, "location": location,
                 "title_parts": [dataset, location]}
                for location in DISPLAY_LOCATIONS for dataset in order
                if any(item_identity(name, item)[0] == dataset for name, item in data.items())]
        paths.append(plot_comparison(data,"Cross-dataset UDA: all datasets",output/"cross_dataset_uda.png",rows))
    elif experiment == "cross_gender":
        for dataset in ("PAMAP2","RealWorld"):
            items={name:item for name,item in data.items() if name.startswith(dataset+"_")}
            groups = ["male", "female"]
            rows = [{"dataset": dataset, "group": group, "location": location,
                     "title_parts": [dataset, group, location]}
                    for location in DISPLAY_LOCATIONS for group in groups
                    if any(item_identity(name, item)[1] == group for name, item in items.items())]
            if items: paths.append(plot_comparison(items,f"Cross-gender UDA: {dataset}",output/f"cross_gender_{dataset}.png",rows))
    else:
        groups = ["below30", "age30plus"]
        rows = [{"dataset": "RealWorld", "group": group, "location": location,
                 "title_parts": ["RealWorld", group, location]}
                for location in DISPLAY_LOCATIONS for group in groups
                if any(item_identity(name, item)[1] == group for name, item in data.items())]
        paths.append(plot_comparison(data,"Cross-age UDA: RealWorld",output/"cross_age_RealWorld.png",rows))
    return paths

if __name__ == "__main__":
    data=choose_and_load(); first=next(iter(data.values()))
    experiment=first["metadata"]["experiment"] if first.get("metadata") else "cross_dataset"
    for path in plot_selected_experiment(data,experiment): print(f"Saved plot: {path}")
