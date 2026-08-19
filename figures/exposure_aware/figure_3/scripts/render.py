from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image, ImageChops, ImageOps
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
PANELS = OUT / "panels"
LEGENDS = OUT / "legends"
S6_OUT = ROOT.parent / "supplementary_figure_6" / "outputs"

def frame(ax):
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color("#222222")
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.grid(False)

def save(fig, path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in cfg["export"]["formats"]:
        fig.savefig(path.with_suffix("." + ext), dpi=cfg["figure"]["dpi"],
                    bbox_inches="tight", transparent=cfg["export"]["transparent"])
    plt.close(fig)

def bar_panel(labels, values, errors, ylabel, ylim, colors, cfg, rotation=42):
    fig, ax = plt.subplots(figsize=cfg["figure"]["panel_size"])
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, color=colors, edgecolor="white", linewidth=0.6,
           capsize=2.5, error_kw={"lw": 0.8, "capthick": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rotation, ha="right")
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim); frame(ax)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.34)
    return fig, ax

def render_ab_distributions(cfg):
    ppb = pd.concat(
        [pd.read_csv(DATA / "ppb_random_split" / f"{split}.csv") for split in ("train", "valid", "test")],
        ignore_index=True,
    )
    panels = [
        ((1 - np.power(10.0, ppb["activity"].to_numpy(float))) * 100,
         np.arange(0, 110, 10), "PPB (%)", "figure_3a_ppb_distribution"),
        (ppb["activity"].to_numpy(float), 10, r"$\log_{10} f_{u}$",
         "figure_3b_unbound_fraction_log10_distribution"),
    ]
    for values, bins, xlabel, stem in panels:
        fig, ax = plt.subplots(figsize=cfg["figure"]["panel_size"])
        ax.hist(values, bins=bins, color=cfg["colors"]["original"], alpha=0.88,
                edgecolor="white", linewidth=0.55)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of compounds")
        frame(ax)
        fig.subplots_adjust(left=0.22, right=0.98, top=0.97, bottom=0.23)
        save(fig, PANELS / stem, cfg)
def render_a(cfg):
    df = pd.read_csv(DATA / "ppb_model_metrics.csv").set_index("model")
    names = ["RF", "XGBoost", "LightGBM", "AttentiveFP", "MolSetRep", "PlasmaBindNet", "PlasmaBindNet-Fu"]
    d = df.loc[names]
    colors = [cfg["colors"]["comparator"]] * 5 + [cfg["colors"]["original"], cfg["colors"]["revised"]]
    fig, _ = bar_panel(names, d.PCC_mean, d.PCC_sd, "Pearson r", (0.55, 0.79), colors, cfg)
    save(fig, PANELS / "figure_3d_ppb_model_comparison", cfg)

def external_selected(metrics, predictions):
    measured = predictions["Measured(PPB%)"].to_numpy(float)
    predicted = predictions["random_full_seed2023_ppb%"].to_numpy(float)
    residual = predicted - measured
    row = {"model": "biasaware_selected_random_seed2023", "PCC": np.corrcoef(measured, predicted)[0,1],
           "RMSE": np.sqrt(np.mean(residual**2))}
    return pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)

def render_bc(cfg):
    metrics = external_selected(pd.read_csv(DATA / "external_ppb_metrics.csv"),
                                pd.read_csv(DATA / "external_ppb_predictions.csv")).set_index("model")
    keys = ["ADMETlab3.0", "admetSAR3.0", "DruMAP", "Pangu Drug Molecule", "PreADMET",
            "OCHEM Consensus model", "paper_MotifAttNet", "biasaware_selected_random_seed2023"]
    labels = ["ADMETlab", "admetSAR", "DruMAP", "Pangu", "PreADMET", "OCHEM", "PlasmaBindNet", "PlasmaBindNet-Fu"]
    colors = [cfg["colors"]["comparator"]] * 6 + [cfg["colors"]["original"], cfg["colors"]["revised"]]
    for panel, metric, ylabel, ylim in [("e", "PCC", "Pearson r", (0.70,1.01))]:
        fig, _ = bar_panel(labels, metrics.loc[keys, metric].to_numpy(float), np.zeros(8), ylabel, ylim, colors, cfg, 50)
        save(fig, PANELS / f"figure_3{panel}_external_{metric.lower()}", cfg)

def render_d(cfg):
    m = pd.read_csv(DATA / "ppb_bin_metrics.csv")
    p = pd.read_csv(DATA / "ppb_bin_paired.csv").query("scheme == 'coarse'").set_index("bin")
    m = m.query("scheme == 'coarse'")
    bins = ["0-40","40-80","80-100"]
    old = m[m.model.str.startswith("Original")].set_index("bin").loc[bins]
    new = m[m.model.str.startswith("New")].set_index("bin").loc[bins]
    fig, ax = plt.subplots(figsize=cfg["figure"]["panel_size"])
    x=np.arange(3); w=.36
    ax.bar(x-w/2, old.mae_pp, w, color=cfg["colors"]["original"])
    ax.bar(x+w/2, new.mae_pp, w, color=cfg["colors"]["revised"])
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={int(old.loc[b,'n'])})" for b in bins])
    ax.set_xlabel("Measured PPB (%)"); ax.set_ylabel("MAE (percentage points)"); ax.set_ylim(0,44)
    for i,b in enumerate(bins):
        delta=float(p.loc[b,"delta_mae_new_minus_old_pp"]); pv=float(p.loc[b,"paired_wilcoxon_p"])
        stars="***" if pv<.001 else "**" if pv<.01 else "*" if pv<.05 else ""
        y=max(old.loc[b,"mae_pp"],new.loc[b,"mae_pp"])+2
        ax.plot([i-w/2,i+w/2],[y,y],color="#333333",lw=.8); ax.text(i,y+.8,f"{delta:+.1f}{stars}",ha="center")
    frame(ax); fig.subplots_adjust(left=.20,right=.98,top=.97,bottom=.25)
    save(fig, PANELS / "figure_3c_ppb_interval_mae", cfg)
    legend = plt.figure(figsize=(4.2,.42)); legend.legend(handles=[Patch(color=cfg["colors"]["original"],label="PlasmaBindNet"),Patch(color=cfg["colors"]["revised"],label="PlasmaBindNet-Fu")],loc="center",ncol=2,frameon=False)
    save(legend, LEGENDS / "figure_3c_legend", cfg)

def render_e(cfg):
    d=pd.read_csv(DATA / "external_ppb_predictions.csv")
    to_neglogfu=lambda ppb: -np.log10(np.clip(1-np.asarray(ppb,dtype=float)/100,1e-4,1))
    measured=to_neglogfu(d["Measured(PPB%)"])
    fu=to_neglogfu(d["random_full_seed2023_ppb%"])
    models=["ADMETlab3.0","admetSAR3.0","DruMAP","Pangu Drug Molecule","PreADMET","OCHEM Consensus model"]
    labels=["ADMETlab","admetSAR","DruMAP","Pangu","PreADMET","OCHEM"]
    fig,axes=plt.subplots(2,3,figsize=(cfg["figure"]["panel_size"][0],1.55),sharex=True,sharey=True)
    for ax,model,label in zip(axes.flat,models,labels):
        predicted=to_neglogfu(d[model])
        for lo,hi,c in [(0,-np.log10(.6),"#EAF2F8"),(-np.log10(.6),-np.log10(.2),"#FFF4D6"),(-np.log10(.2),4,"#F9E6E3")]: ax.axvspan(lo,hi,color=c,zorder=0)
        ax.plot([0,4],[0,4],":",color="#333333",lw=.42)
        ax.scatter(measured,predicted,s=8.5,facecolor="white",edgecolor=cfg["colors"]["original"],linewidth=.55,alpha=.92)
        ax.scatter(measured,fu,s=7.0,color=cfg["colors"]["revised"],edgecolor="white",linewidth=.20,alpha=.78)
        ax.set(xlim=(-.08,4.08),ylim=(-.08,4.08),xticks=(0,2,4),yticks=(0,2,4))
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(.45); spine.set_color("#222222")
        ax.tick_params(direction="out",length=1.5,width=.45,labelsize=cfg["text"]["tick_size"],pad=1.0); ax.grid(False)
        ax.text(.055,.93,label,transform=ax.transAxes,ha="left",va="top",fontsize=5.7,fontweight="bold")
    fig.supxlabel(r"Measured $-\log_{10} f_u$",fontsize=cfg["text"]["label_size"],y=.005)
    fig.supylabel(r"Predicted $-\log_{10} f_u$",fontsize=cfg["text"]["label_size"],x=.008)
    fig.subplots_adjust(left=.145,right=.995,top=.995,bottom=.19,wspace=.16,hspace=.12); save(fig,PANELS/"figure_3f_ppb_scatter",cfg)
    handles=[Line2D([],[],marker="o",ls="none",mfc="white",mec=cfg["colors"]["original"],label="Online model"),Line2D([],[],marker="o",ls="none",mfc=cfg["colors"]["revised"],mec="white",label="PlasmaBindNet-Fu"),Patch(facecolor="#EAF2F8",edgecolor="#B8C4CC",label="PPB 0–40%"),Patch(facecolor="#FFF4D6",edgecolor="#D7C99A",label="PPB 40–80%"),Patch(facecolor="#F9E6E3",edgecolor="#D9B7B2",label="PPB 80–100%")]
    legend=plt.figure(figsize=(6.8,.42)); legend.legend(handles=handles,loc="center",ncol=5,frameon=False,columnspacing=1.1,handletextpad=.45)
    save(legend,LEGENDS/"figure_3f_legend",cfg)

def render_f(cfg):
    labels=["RF","XGBoost","LightGBM","DoseExpoNet"]; vals=[.7542,.7406,.7659,.8032]; err=[.0171,.0292,.0056,.0057]
    colors=[cfg["colors"]["comparator"]]*3+[cfg["colors"]["original"]]
    fig,_=bar_panel(labels,vals,err,"Pearson r",(.69,.84),colors,cfg,32); save(fig,PANELS/"figure_3g_cmax_model_comparison",cfg)

def render_case(case, panel, cfg):
    d=pd.read_csv(DATA/"cmax_case_predictions.csv").query("case_name == @case").sort_values("dose")
    p=pd.read_csv(DATA/"cmax_case_ppb_summary.csv").set_index("case_name").loc[case]
    fu=1-float(p.plasmabindnet_fu_ppb_percent_mean)/100; measured_fu=1-float(p.measured_ppb_percent)/100
    x=d.dose.to_numpy(float); pred=d.predicted_activity_mean.to_numpy(float); sd=d.predicted_activity_sd.to_numpy(float); obs=d.activity.to_numpy(float)
    fig,ax=plt.subplots(figsize=cfg["figure"]["panel_size"])
    ax.fill_between(x,pred-sd,pred+sd,color=cfg["colors"]["interval"],alpha=.55,lw=0)
    ax.plot(x,pred,color=cfg["colors"]["original"],lw=1.8); ax.scatter(x,obs,s=24,color=cfg["colors"]["original"],edgecolor="white",linewidth=.55,zorder=4)
    ax.plot(x,pred+np.log10(fu),color=cfg["colors"]["revised"],lw=1.8,ls="--")
    ax.scatter(x,obs+np.log10(measured_fu),s=23,marker="D",facecolor="white",edgecolor=cfg["colors"]["revised"],linewidth=.8,zorder=4)
    ax.set_xscale("log",base=2); ax.set_xlabel("Dose (mg)"); ax.set_ylabel(r"$\log_{10}$ concentration ($\mu$g mL$^{-1}$)")
    ax.text(.96,.06,case,transform=ax.transAxes,ha="right",va="bottom",fontsize=cfg["text"]["drug_name_size"],fontweight="bold")
    frame(ax); fig.subplots_adjust(left=.20,right=.98,top=.97,bottom=.22); save(fig,PANELS/f"figure_3{panel}_{case.split()[0].lower()}",cfg)

def render_case_legend(cfg):
    handles=[Line2D([],[],color=cfg["colors"]["original"],lw=1.8,label="Predicted total Cmax"),Line2D([],[],marker="o",ls="none",mfc=cfg["colors"]["original"],mec="white",label="Observed total Cmax"),Line2D([],[],color=cfg["colors"]["revised"],lw=1.8,ls="--",label="Predicted free Cmax"),Line2D([],[],marker="D",ls="none",mfc="white",mec=cfg["colors"]["revised"],label="Observed free Cmax")]
    fig=plt.figure(figsize=(6.8,.55)); fig.legend(handles=handles,loc="center",ncol=4,frameon=False,columnspacing=1.0,handlelength=1.8); save(fig,LEGENDS/"figure_3h_i_legend",cfg)

def trim(path):
    im=Image.open(path).convert("RGB"); diff=ImageChops.difference(im,Image.new("RGB",im.size,"white")).convert("L"); box=diff.point(lambda v:255 if v>8 else 0).getbbox(); return im.crop(box) if box else im

def import_s6_distribution_panels(cfg):
    sources = {
        "figure_3a_ppb_distribution": "supplementary_figure_6a_ppb_distribution",
        "figure_3b_unbound_fraction_log10_distribution": "supplementary_figure_6b_unbound_fraction_log10_distribution",
    }
    for destination, source in sources.items():
        for ext in cfg["export"]["formats"]:
            src = S6_OUT / f"{source}.{ext}"
            dst = PANELS / f"{destination}.{ext}"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)

def composite(cfg):
    names=["figure_3a_ppb_distribution","figure_3b_unbound_fraction_log10_distribution","figure_3c_ppb_interval_mae","figure_3d_ppb_model_comparison","figure_3e_external_pcc","figure_3f_ppb_scatter","figure_3g_cmax_model_comparison","figure_3h_cevimeline","figure_3i_ciprofloxacin"]
    imgs=[trim(PANELS/(n+".png")) for n in names]
    fig,axes=plt.subplots(3,3,figsize=cfg["figure"]["composite_size"]); axes=axes.flat
    for panel_index, (ax, im) in enumerate(zip(axes, imgs)):
        ax.imshow(im)
        ax.axis("off")
        ax.text(-0.025, 1.015, chr(ord("a") + panel_index), transform=ax.transAxes,
                ha="left", va="top", fontsize=10, fontweight="bold", clip_on=False)
    fig.subplots_adjust(left=.005,right=.995,top=.995,bottom=.005,wspace=cfg["layout"]["wspace"],hspace=cfg["layout"]["hspace"])
    fig.savefig(OUT/"figure_3.png",dpi=cfg["figure"]["dpi"],bbox_inches="tight"); fig.savefig(OUT/"figure_3.pdf",dpi=cfg["figure"]["dpi"],bbox_inches="tight"); plt.close(fig)

def main():
    cfg=yaml.safe_load((ROOT/"params.yaml").read_text(encoding="utf-8"))
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":[cfg["text"]["font_family"],"Helvetica","DejaVu Sans"],"font.size":cfg["text"]["font_size"],"axes.labelsize":cfg["text"]["label_size"],"xtick.labelsize":cfg["text"]["tick_size"],"ytick.labelsize":cfg["text"]["tick_size"],"pdf.fonttype":42,"svg.fonttype":"none"})
    PANELS.mkdir(parents=True,exist_ok=True); LEGENDS.mkdir(parents=True,exist_ok=True)
    render_ab_distributions(cfg)
    render_a(cfg); render_bc(cfg); render_d(cfg); render_e(cfg); render_f(cfg)
    render_case("Ciprofloxacin hydrochloride","i",cfg); render_case("Cevimeline hydrochloride","h",cfg); render_case_legend(cfg); composite(cfg)

if __name__ == "__main__": main()
