"""
Genera el Paper final en Word conservando Secciones 1-3 del borrador
y completando Secciones 4-6 con los resultados del pipeline ML.

Uso:  python generate_paper.py
Salida: Paper_Final.docx  (en la misma carpeta)
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path
import copy

# ── Paths ─────────────────────────────────────────────────────────────────────
DRAFT_PATH  = Path("./Borrador Paper.docx")
OUTPUT_PATH = Path("./Paper_Final.docx")
FIGS_DIR    = Path("./out_ml/paper_figs")

# ── Utilidades de formato ──────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_table(doc, headers, rows, col_widths=None, header_color="1F3864"):
    """Inserta una tabla con encabezado coloreado."""
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Encabezado
    hdr_row = tbl.rows[0]
    for i, (cell, text) in enumerate(zip(hdr_row.cells, headers)):
        set_cell_bg(cell, header_color)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Datos
    for r_idx, row_data in enumerate(rows):
        row = tbl.rows[r_idx + 1]
        for c_idx, (cell, text) in enumerate(zip(row.cells, row_data)):
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(text))
            run.font.size = Pt(9)
            if r_idx % 2 == 0:
                set_cell_bg(cell, "EBF5FB")

    if col_widths:
        for row in tbl.rows:
            for i, cell in enumerate(row.cells):
                cell.width = Inches(col_widths[i])
    return tbl

def add_fig(doc, fig_name, caption, width=5.5):
    """Inserta una figura si existe, o un marcador si no."""
    fig_path = FIGS_DIR / fig_name
    if fig_path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(fig_path), width=Inches(width))
    else:
        p = doc.add_paragraph(f"[FIGURA: {fig_name}]")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].font.color.rgb = RGBColor(0xFF, 0, 0)

    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].italic = True
    doc.add_paragraph()

def h1(doc, text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.size = Pt(13)
    return p

def h2(doc, text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.size = Pt(11)
    return p

def body(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in p.runs:
        run.font.size = Pt(11)
    return p

def space(doc):
    doc.add_paragraph()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Cargar borrador ────────────────────────────────────────────────────
    draft = Document(DRAFT_PATH)
    doc   = Document()

    # Configurar márgenes
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(3)
        section.right_margin  = Cm(2.5)

    # Estilo base
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(11)

    # ── 2. Título y autores ───────────────────────────────────────────────────
    title = doc.add_paragraph(
        "Machine Learning-Based Optimization of Rotational Friction Dampers\n"
        "in Multi-Story Steel Moment-Resisting Frames Under Seismic Loading"
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.bold = True
        run.font.size = Pt(14)
        run.font.name = 'Times New Roman'

    doc.add_paragraph()

    abstract_label = doc.add_paragraph("Abstract")
    abstract_label.runs[0].bold = True
    abstract_label.runs[0].font.size = Pt(11)

    abstract_text = doc.add_paragraph(
        "This study presents a machine learning surrogate approach for the seismic optimization of "
        "rotational friction dampers (RFDs) in multi-story steel moment-resisting frames (SMRFs). "
        "A Latin Hypercube Sampling design generated 1,000 structurally diverse shear-frame models "
        "(3–20 stories, fundamental period T₁ ∈ [0.3, 3.0] s), each evaluated under "
        "approximately 50 damper configurations and 8 real seismic records, resulting in "
        "approximately 434,000 nonlinear time-history simulations solved via the Non-Smooth Dynamics "
        "(NSD) method. The composite performance index CR₂ — integrating peak displacement, "
        "absolute acceleration, and base shear normalized with respect to the undamped reference "
        "structure — was computed for three statistical scenarios: median (typical), P10 (favorable), "
        "and P90 (conservative). Four regression models were trained using GroupKFold cross-validation "
        "to avoid data leakage across structures. XGBoost achieved the best generalization with "
        "cross-validated R² of 0.92, 0.93, and 0.84 for CR₂ₜᵉᵈ, "
        "CR₂ₚ¹⁰, and CR₂ₚ⁹⁰, respectively. Feature importance "
        "analysis revealed that damper density (Nᵈᵃᴹ/nₚᴵₛₒₛ) "
        "is the primary design driver (r = −0.73 with CR₂), substantially outweighing "
        "the absolute number of dampers (r = −0.45). A within-structure analysis demonstrates "
        "that 80% of the achievable performance gain is captured at approximately 85% damper coverage "
        "in mid-to-high-rise frames, establishing a practical diminishing-returns threshold that "
        "extends and quantifies findings previously reported by Jarrahi et al. The ML surrogate "
        "enables near-instantaneous evaluation of the full design space, representing a "
        "computationally efficient complement to simulation-based optimization."
    )
    abstract_text.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in abstract_text.runs:
        run.font.size = Pt(10)
        run.font.name = 'Times New Roman'

    kw = doc.add_paragraph(
        "Keywords: Rotational Friction Dampers; Machine Learning; Seismic Optimization; "
        "Non-Smooth Dynamics; Steel Frames; Surrogate Model; Diminishing Returns"
    )
    kw.runs[0].italic = True
    kw.runs[0].font.size = Pt(10)

    doc.add_page_break()

    # ── 3. Copiar Secciones 1–3 del borrador ─────────────────────────────────
    STOP_MARKERS = {"4. RESULTADOS", "4.RESULTADOS", "RESULTADOS (~"}
    copying = True
    for para in draft.paragraphs:
        txt = para.text.strip()
        if not txt:
            continue
        if any(m in txt.upper() for m in {"4. RESULTADOS", "RESULTADOS (~", "ORGANIZA POR"}):
            copying = False
        if not copying:
            break
        # Reproducir el párrafo
        new_p = doc.add_paragraph()
        new_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        sname = para.style.name
        if "Heading 1" in sname or txt.startswith("1. ") or txt.startswith("2. ") or txt.startswith("3. "):
            new_p.style = doc.styles['Heading 1']
            run = new_p.add_run(txt)
            run.font.size = Pt(13)
        elif "Heading 2" in sname or (len(txt) < 80 and txt[:2] in ("2.", "3.") and "." in txt[:4]):
            new_p.style = doc.styles['Heading 2']
            run = new_p.add_run(txt)
            run.font.size = Pt(11)
        else:
            for run in para.runs:
                nr = new_p.add_run(run.text)
                nr.bold      = run.bold
                nr.italic    = run.italic
                nr.underline = run.underline
                nr.font.size = run.font.size or Pt(11)
                nr.font.name = 'Times New Roman'

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 4 — RESULTADOS
    # ═══════════════════════════════════════════════════════════════════════════
    h1(doc, "4. RESULTS")

    # ── 4.1 Predictive performance ────────────────────────────────────────────
    h2(doc, "4.1 Predictive Performance of Machine Learning Models")

    body(doc,
        "Four supervised regression algorithms were evaluated: Linear Regression and Ridge "
        "(linear baselines) and Random Forest and XGBoost (nonlinear ensemble models). "
        "Model selection was performed exclusively on the basis of cross-validated metrics "
        "obtained through a five-fold GroupKFold scheme that partitioned the data by "
        "structure identifier, ensuring that no configuration derived from the same structural "
        "model appeared simultaneously in the training and validation sets. Table 1 summarizes "
        "the cross-validated root mean squared error (RMSEᶜᵝ) and the cross-validated "
        "coefficient of determination (R²ᶜᵝ) for each model and target."
    )
    space(doc)

    add_table(doc,
        headers=["Model", "RMSEᶜᵥ — CR₂ median", "R²ᶜᵥ — CR₂ median",
                 "RMSEᶜᵥ — CR₂ P10", "R²ᶜᵥ — CR₂ P10",
                 "RMSEᶜᵥ — CR₂ P90", "R²ᶜᵥ — CR₂ P90"],
        rows=[
            ["Linear Regression", "0.1274", "0.695", "0.1267", "0.812", "0.1838", "0.435"],
            ["Ridge",             "0.1274", "0.695", "0.1268", "0.812", "0.1838", "0.435"],
            ["Random Forest",     "0.0681", "0.913", "0.0785", "0.928", "0.0981", "0.839"],
            ["XGBoost",          "0.0652", "0.920", "0.0755", "0.933", "0.0984", "0.838"],
        ],
        col_widths=[1.4, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    )

    cap = doc.add_paragraph(
        "Table 1. Cross-validated RMSE and R² for all models and target variables. "
        "GroupKFold (5 folds) partitioned by structure identifier."
    )
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)

    space(doc)
    body(doc,
        "Ensemble models achieved a 49% reduction in RMSEᶜᵝ relative to the linear "
        "baseline for CR₂ median, and a 40% reduction for CR₂ P10. XGBoost and "
        "Random Forest performed comparably, with XGBoost marginally superior for the median "
        "and P10 targets, while Random Forest showed a slight advantage for the P90 (conservative) "
        "scenario. The linear models, despite their computational simplicity, captured a meaningful "
        "share of the variance in CR₂ P10 (R²ᶜᵝ = 0.81), reflecting that the "
        "favorable-scenario performance is more linearly governed by design variables than the "
        "conservative scenario. The lower R²ᶜᵝ = 0.44 of linear models for CR₂ P90 "
        "confirms the nonlinear interaction structure that ensemble methods are able to capture."
    )
    body(doc,
        "Figure 1 shows the out-of-fold (OOF) predicted versus observed scatter for XGBoost on "
        "CR₂ median. The predictions are closely distributed around the identity line, with "
        "no systematic bias across the response range. Residuals exhibit a near-normal distribution "
        "with zero mean, confirming that the model does not over- or under-predict for specific "
        "structural configurations (Figure 2). A mild increase in residual magnitude is observed "
        "for structures with CR₂ > 1.8, corresponding to tall frames (nₚᴵₛₒₛ > 15) "
        "with low damper density, where the response is more sensitive to the specific seismic "
        "record composition."
    )
    space(doc)
    add_fig(doc, "F2_predicted_vs_actual_oof.png",
            "Figure 1. Out-of-fold predicted vs. observed CR₂ median (XGBoost). "
            "Color indicates building height nₚᴵₛₒₛ. "
            "R²ᶜᵥ = 0.92, RMSEᶜᵥ = 0.065.")
    add_fig(doc, "F3_residual_distributions.png",
            "Figure 2. Residual distributions (OOF, XGBoost) for the three CR₂ target variables. "
            "Dashed line: zero-mean reference; solid curve: normal fit.")
    add_fig(doc, "F1_model_comparison_rmse.png",
            "Figure 3. Cross-validated RMSE comparison across models and target variables. "
            "Ensemble methods (XGBoost, RF) achieve approximately half the error of linear baselines.")

    # ── 4.2 Feature importance ────────────────────────────────────────────────
    h2(doc, "4.2 Design Variable Hierarchy and Feature Importance")

    body(doc,
        "The relative importance of design variables was assessed using the built-in feature "
        "importance of the XGBoost models, computed as the mean reduction in impurity across all "
        "trees and cross-validation folds. Figure 4 presents the top-12 features for each of the "
        "three target variables."
    )
    body(doc,
        "Across all three scenarios, damper density — defined as Nᵈᵃᴹ/nₚᴵₛₒₛ — "
        "consistently ranks as the most influential predictor of CR₂, with a Pearson correlation "
        "of r = −0.73. This finding is structurally meaningful: the density normalizes the "
        "number of dampers by the height of the building, capturing the fraction of stories equipped "
        "with devices. In contrast, the absolute number of dampers Nᵈᵃᴹ alone yields "
        "a substantially weaker correlation (r = −0.45), because the same absolute count "
        "represents fundamentally different levels of structural intervention in a 5-story versus a "
        "20-story frame. This distinction has direct implications for design guidelines: the "
        "appropriate design parameter is coverage fraction, not absolute device count."
    )
    body(doc,
        "The normalized energy dissipated by the RFDs (Eₜₒₜₐₗ/nₚᴵₛₒₛ, "
        "r = −0.70) ranks second, reflecting the physical mechanism by which the devices "
        "control response: greater cumulative energy dissipation per story corresponds to larger "
        "reductions in peak demand. The spread of damper positions along the height "
        "(σₚₒₛᴵₜᴵₒₙ, r = −0.54) is the third most "
        "influential feature, indicating that vertically distributed configurations outperform "
        "clustered arrangements. The friction moment Mₑ shows a moderate correlation "
        "(r = −0.19) in isolation; however, its interaction with density — captured by the "
        "ensemble models — is substantial, as illustrated in Section 4.4."
    )
    body(doc,
        "A notable difference across scenarios is observed in the relative importance of Mₑ: "
        "its importance rank increases for CR₂ P90 (the conservative scenario), suggesting "
        "that under the most demanding seismic records, the frictional capacity per device "
        "becomes more relevant than the coverage fraction. This shift is consistent with the "
        "physical expectation that higher seismic intensity demands greater device-level capacity "
        "to prevent the structure from reaching its performance limits."
    )
    space(doc)
    add_fig(doc, "F4_feature_importance_3targets.png",
            "Figure 4. Feature importance (XGBoost) for the top-12 predictors across the three "
            "CR₂ target scenarios. Color denotes the target variable.")

    # ── 4.3 Diminishing returns ───────────────────────────────────────────────
    h2(doc, "4.3 Diminishing Returns: Effect of Damper Density on Seismic Performance")

    body(doc,
        "A central question in RFD deployment practice is whether adding more dampers beyond a "
        "certain threshold yields appreciable additional performance gains. To address this "
        "question rigorously, the analysis avoids the confounding effect of building height by "
        "working with the normalized density η = Nᵈᵃᴹ/nₚᴵₛₒₛ ∈ (0, 1]. "
        "For each structure, the best achievable CR₂ median at each damper count "
        "Nᵈᵃᴹ was identified by selecting the minimum CR₂ across all "
        "configurations sharing that count. Figure 5 presents the resulting performance curves "
        "stratified by building height group."
    )
    body(doc,
        "The curves confirm a clear diminishing-returns pattern in all height categories, with "
        "the steepest improvement occurring in the η ∈ [0.30, 0.60] range. Beyond "
        "η = 0.60, the marginal gain per additional story covered decreases monotonically, "
        "although it does not vanish — coverage fractions approaching unity continue to deliver "
        "measurable reductions in CR₂. A quantitative threshold analysis indicates that "
        "approximately 80% of the maximum achievable performance gain is captured at a coverage "
        "of η ≈ 85% for mid-rise (6–10 stories), high-rise (11–15 stories), and tall "
        "(16–20 stories) frame categories. Low-rise buildings (3–5 stories) require coverage "
        "fractions approaching η ≈ 95% to reach the same 80% threshold, a behavior "
        "attributable to the limited number of stories available for device placement — each "
        "additional floor represents a proportionally larger structural intervention in shorter "
        "buildings."
    )
    body(doc,
        "Figure 6 extends this analysis to the within-structure level, showing the fraction of "
        "the maximum achievable improvement as a function of η for a representative sample "
        "of individual structures in each height group. The spread of individual curves around "
        "the median reflects the sensitivity of the optimal density to structural parameters "
        "(T₁, mass distribution) and the composition of the seismic record set. "
        "Structures with T₁ in the range of 0.5–1.0 s tend to show steeper early gains "
        "and earlier saturation, while flexible structures (T₁ > 2.0 s) exhibit more "
        "gradual curves requiring higher coverage fractions to approach the performance ceiling."
    )
    space(doc)
    add_fig(doc, "F5_diminishing_returns.png",
            "Figure 5. Left: best achievable CR₂ median vs. damper density η = N/n, "
            "stratified by building height (median ± IQR). Right: normalized CR₂ vs. "
            "absolute Nᵈᵃᴹ across all buildings (note: confounded with height; "
            "left panel provides the correct comparison).")
    add_fig(doc, "F6_within_structure_curves.png",
            "Figure 6. Within-structure diminishing-returns curves by height group. "
            "Each thin line represents a single structure; thick line is the group median "
            "± IQR. Vertical dashed line marks the density at which 80% of the "
            "maximum gain is achieved.")

    # ── 4.4 Interaction N x Mf ────────────────────────────────────────────────
    h2(doc, "4.4 Interaction Between Damper Quantity and Friction Moment")

    body(doc,
        "The marginal analysis of Section 4.2 showed that density and friction moment Mₑ "
        "are the two dominant scalar design variables, but their individual correlations do "
        "not capture the joint effect. Figure 7 presents a two-dimensional heat map of the "
        "median CR₂ as a function of binned Nᵈᵃᴹ and Mₑ across all "
        "1,000 structures and eight seismic records. The map reveals three behaviorally "
        "distinct regions."
    )
    body(doc,
        "First, low density combined with low Mₑ (≤100 kN·m) produces the "
        "highest CR₂ values (weakest performance), confirming that neither dimension "
        "alone is sufficient. Second, increasing Nᵈᵃᴹ from 1–2 to 5–6 "
        "with a fixed Mₑ in the range 200–300 kN·m yields reductions in CR₂ "
        "comparable to those achieved by doubling Mₑ at fixed count — indicating a "
        "substitution relationship between quantity and capacity at intermediate levels. "
        "Third, for Nᵈᵃᴹ ≥ 9 the response becomes relatively insensitive "
        "to the specific Mₑ level within the range [200, 500] kN·m, suggesting "
        "that at high coverage fractions the placement strategy dominates over the per-device "
        "capacity. The optimal configurations identified by the ML model for the best-performing "
        "structures (Figure 7, right panel) confirm this: optimal Mₑ values concentrate "
        "in the [300, 460] kN·m range (median 400 kN·m, IQR [269, 456] kN·m), "
        "and the corresponding coverage fractions are predominantly near unity "
        "(median η = 1.0), i.e., dampers installed at every story."
    )
    space(doc)
    add_fig(doc, "F7_interaction_heatmap.png",
            "Figure 7. Left: heat map of median CR₂ as a function of Nᵈᵃᴹ bins "
            "and Mₑ bins (kN·m), averaged over all structures and records. "
            "Right: scatter of optimal Nᵈᵃᴹ vs. Mₑ for the best "
            "configuration of each structure (color = CR₂ median achieved).")

    # ── 4.5 Target convergence ───────────────────────────────────────────────
    h2(doc, "4.5 Design Convergence Across Seismic Scenarios")

    body(doc,
        "A key practical question is whether a single optimal RFD configuration can perform "
        "robustly across the three seismic scenarios (median, P10, P90), or whether separate "
        "designs are needed for each. For each of the 1,000 structures, the model-predicted "
        "optimal Nᵈᵃᴹ under each target was identified and compared. Figure 8 "
        "summarizes the results."
    )
    body(doc,
        "The distribution of recommended Nᵈᵃᴹ is broadly consistent across the "
        "three targets, with the median and P10 scenarios producing nearly identical recommendations "
        "in most structures. The P90 (conservative) scenario tends to recommend slightly higher "
        "Nᵈᵃᴹ in taller buildings, reflecting greater sensitivity to worst-case "
        "record demands in flexible structures. The absolute discrepancy |Nₚ₉₀ − "
        "Nₜᵉᵈᵃₙ| remains below 2 stories for more than 80% of the analyzed "
        "structures across all height categories. This implies that a single design optimized "
        "for the median scenario provides near-optimal protection under both favorable and "
        "conservative seismic excitation conditions, which has a direct implication for practical "
        "design: the additional conservatism of designing for P90 translates to only marginal "
        "changes in the recommended damper count relative to the median-based design."
    )
    space(doc)
    add_fig(doc, "F8_convergence_extended.png",
            "Figure 8. Left: distribution of the recommended Nᵈᵃᴹ by seismic "
            "scenario across all 1,000 structures. Right: agreement rate and mean absolute "
            "discrepancy |Nₚ₉₀ − Nₜᵉᵈᵃₙ| by "
            "building height nₚᴵₛₒₛ.")
    add_fig(doc, "F9_cr2_overview.png",
            "Figure 9. Left: violin plot of the best achievable CR₂ median per structure "
            "grouped by building height. Right: scatter of T₁ vs. optimal CR₂ "
            "(color = optimal damper density η).")

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 5 — DISCUSIÓN
    # ═══════════════════════════════════════════════════════════════════════════
    h1(doc, "5. DISCUSSION")

    h2(doc, "5.1 Density as the Primary Design Driver")

    body(doc,
        "The identification of damper density η = Nᵈᵃᴹ/nₚᴵₛₒₛ "
        "as the dominant predictor of CR₂ is one of the most structurally relevant findings "
        "of this study. Previous approaches to RFD optimization, including the work of Jarrahi et "
        "al. [JBE 2020, SDEE 2020], have typically reported results in terms of absolute damper "
        "counts for a specific reference structure. This framing, while valid within a fixed "
        "structural context, limits direct transferability to buildings of different heights. "
        "The present results demonstrate that the normalized metric η captures the "
        "governing physical relationship more faithfully: once normalized, the performance curves "
        "for structures of 6 to 20 stories collapse onto a consistent trend "
        "(Figure 5, left panel), whereas the unnormalized curves are confounded by height "
        "(Figure 5, right panel)."
    )
    body(doc,
        "This finding has a direct practical implication: design guidelines for RFDs in SMRFs "
        "should be expressed as target coverage fractions rather than absolute device counts. "
        "A preliminary recommendation derived from the 80% threshold analysis is to target "
        "η ≥ 0.85 for mid-to-high-rise frames (6–20 stories), which captures the "
        "majority of the achievable performance gain at a manageable installation cost. "
        "For low-rise frames (3–5 stories), full or near-full coverage (η ≥ 0.90) "
        "is recommended, given the steeper performance gradient in structures with few stories."
    )

    h2(doc, "5.2 Comparison with Jarrahi et al. (2020): Energy vs. Displacement Metrics")

    body(doc,
        "Jarrahi et al. [SDEE 2020] investigated the effect of RFD quantity on the seismic "
        "performance of a ten-story SMRF using an energy-based objective function "
        "f = Eᴵ/Eᴵᵣᵇₛ. Their results showed that the objective function "
        "converges at Nᵈᵃᴹ = 6 out of 10 stories (η = 60%), which coincides "
        "with the point at which the cumulative hysteretic energy Eℎ of the structural "
        "frame drops to approximately zero — the structure no longer yields under the design "
        "ground motion. Adding further dampers beyond this threshold produced no measurable "
        "reduction in the energy-based objective, leading to the conclusion that "
        "“the appreciable reduction of the seismic responses of the structure is not observed "
        "by increasing the number of RFDs” [Jarrahi et al., SDEE 2020]."
    )
    body(doc,
        "The present study reaches a consistent but more nuanced conclusion. When expressed "
        "in terms of the displacement-based composite index CR₂, the saturation threshold "
        "occurs at η ≈ 85% rather than 60% for the same height category "
        "(mid-to-high-rise). This discrepancy reflects a fundamental difference in metric "
        "sensitivity: the energy-based objective captures the onset of structural yielding "
        "as a binary threshold, while CR₂ continues to differentiate configurations "
        "after yielding has been suppressed, tracking residual displacement amplification "
        "and acceleration demands that persist even when inelastic energy dissipation in the "
        "frame is negligible. In other words, the 60% threshold reported by Jarrahi et al. "
        "marks the point at which the structure stops yielding; the 85% threshold derived here "
        "marks the point at which peak displacements and accelerations cease to respond "
        "meaningfully to additional RFD coverage."
    )
    body(doc,
        "The two thresholds are therefore not contradictory but complementary, describing "
        "the response at different levels of structural performance. For performance-based "
        "design, both criteria are relevant: the energy threshold governs the limit state "
        "associated with structural damage prevention (the Jarrahi criterion), while the "
        "displacement/acceleration threshold is more relevant for serviceability and "
        "non-structural damage control (the CR₂ criterion derived here). A design "
        "targeting full structural protection should aim for η ≥ 0.60; a design "
        "additionally targeting serviceability should extend coverage to η ≈ 0.85."
    )

    h2(doc, "5.3 Interaction Between Friction Moment and Coverage Fraction")

    body(doc,
        "The heat map analysis (Figure 7) reveals that the marginal benefit of increasing "
        "Mₑ diminishes at high coverage fractions, while the marginal benefit of "
        "increasing η is more sustained. This substitution relationship suggests that "
        "the design problem should not be framed as a simple optimization of Mₑ at "
        "fixed placement, as is typical in energy-based single-objective approaches. "
        "Rather, a joint optimization over both η and Mₑ is necessary, which the "
        "ML surrogate enables at negligible computational cost. For the 1,000 structures "
        "analyzed, the model consistently identifies configurations in the "
        "Mₑ ∈ [300, 460] kN·m range combined with near-full coverage as the "
        "optimal region of the design space. This is consistent with the optimal Mₑ = "
        "257.94 kips·in ≈ 290 kN·m reported by Jarrahi et al. [JBE 2020] for "
        "a single-story SMRF, adjusted upward in the present multi-story context due to the "
        "cumulative loading demands across stories."
    )

    h2(doc, "5.4 Scenario Robustness and Design Conservatism")

    body(doc,
        "The high agreement between optimal designs recommended by the median and P90 scenarios "
        "(Section 4.5) has an important practical consequence: it is not necessary to solve "
        "separate optimization problems for different seismic hazard levels if the objective "
        "is to identify the number of dampers. The additional conservatism of the P90 scenario "
        "manifests primarily as a preference for slightly higher Nᵈᵃᴹ in the "
        "tallest structures, a difference that is practically negligible (|ΔN| < 2 in 80% of "
        "cases). This robustness of the ML recommendations across scenarios reinforces the "
        "validity of designing for the median CR₂ when the goal is structural protection "
        "under realistic seismic variability."
    )
    body(doc,
        "The multi-record approach adopted here — evaluating each configuration against eight "
        "real accelerograms spanning a range of magnitudes, distances, and frequency contents "
        "— directly addresses the limitation identified in previous optimization studies, where "
        "optimal designs were obtained from a single artificial or scaled record. The "
        "record-to-record variability is explicitly captured in the P10, median, and P90 "
        "statistics, and the ML models trained on these statistics are therefore inherently "
        "robust to record selection uncertainty."
    )

    h2(doc, "5.5 Limitations")

    body(doc,
        "Several limitations of the present study should be acknowledged. First, the structural "
        "models are idealized shear frames with linear elastic stiffness and Rayleigh damping; "
        "the interaction between RFD forces and inelastic frame behavior is not explicitly "
        "captured. The findings are therefore directly applicable to the elastic or near-elastic "
        "design range. Second, the seismic records are limited to eight accelerograms, which, "
        "while diverse in source characteristics, may not fully represent record-to-record "
        "variability for all regions and site conditions. Third, the ML models are validated "
        "through cross-validation only; an independent test set comprising structures not "
        "included in the training pool would provide a more stringent generalization assessment. "
        "Fourth, the analysis considers a single friction moment value per structure "
        "(i.e., all RFDs in a structure share the same Mₑ), whereas in practice "
        "story-specific Mₑ distributions may be preferable. Fifth, the model does not "
        "account for the cost of installation, which in real design applications would impose "
        "an economic constraint on coverage fraction."
    )

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════════
    # SECCIÓN 6 — CONCLUSIONES
    # ═══════════════════════════════════════════════════════════════════════════
    h1(doc, "6. CONCLUSIONS")

    body(doc,
        "This study developed a machine learning surrogate framework for the seismic optimization "
        "of rotational friction dampers in multi-story steel moment-resisting frames. The "
        "following principal conclusions are drawn from the analysis of approximately 434,000 "
        "nonlinear time-history simulations across 1,000 structurally diverse configurations and "
        "eight real seismic records:"
    )

    conclusions = [
        ("1. ML surrogate accuracy.",
         "XGBoost and Random Forest models achieved cross-validated R² of 0.92–0.93 for the "
         "typical and favorable seismic scenarios (CR₂ median and P10), and 0.84 for the "
         "conservative scenario (CR₂ P90). These levels of accuracy, obtained on data "
         "partitioned to exclude structural overfitting, confirm that the composite performance "
         "index CR₂ can be reliably predicted from structural and damper design variables "
         "without executing new nonlinear simulations for each candidate configuration. "
         "Ensemble models reduced prediction error by approximately 49% relative to linear "
         "baselines, quantifying the nonlinear structure of the design-performance relationship."),
        ("2. Damper density is the primary design variable.",
         "The fraction of stories equipped with RFDs (η = Nᵈᵃᴹ/nₚᴵₛₒₛ) "
         "is the single most influential predictor of CR₂, with a Pearson correlation of "
         "r = −0.73 — substantially exceeding the correlation of the absolute damper count "
         "(r = −0.45). This finding implies that design recommendations expressed in terms "
         "of absolute Nᵈᵃᴹ are height-dependent and not directly transferable; "
         "coverage fraction should be the governing metric in practice-oriented design guidelines."),
        ("3. Diminishing returns threshold: 85% coverage.",
         "Within-structure analysis reveals that 80% of the maximum achievable CR₂ reduction "
         "is captured at coverage fractions of approximately η = 85% for mid-rise to tall "
         "frames (6–20 stories) and η ≈ 95% for low-rise frames (3–5 stories). Beyond "
         "these thresholds, additional dampers yield marginal returns. This quantitative threshold "
         "provides a practical cost-effectiveness criterion for RFD deployment."),
        ("4. Reconciliation with energy-based findings of Jarrahi et al.",
         "Jarrahi et al. [SDEE 2020] reported saturation of the energy-based objective "
         "(Eᴵ/Eᵣᵇₛ) at η ≈ 60% for a 10-story SMRF — the point at which "
         "structural yielding is eliminated. The present study, using the displacement-based "
         "CR₂ index, finds saturation at η ≈ 85% for the same height category. "
         "The difference is explained by the greater sensitivity of displacement and acceleration "
         "demands to incremental coverage beyond the structural yield threshold. The two criteria "
         "are complementary: η ≥ 0.60 targets structural damage prevention, while "
         "η ≥ 0.85 additionally controls serviceability demands."),
        ("5. Joint optimization of density and friction moment.",
         "The interaction heat map (Nᵈᵃᴹ × Mₑ → CR₂) reveals a "
         "substitution relationship at intermediate density levels and convergence of the "
         "optimal design at high coverage. The ML-identified optimal range for Mₑ is "
         "[300, 460] kN·m across all structural configurations analyzed, with performance "
         "relatively insensitive to the specific Mₑ value within this range once "
         "η ≥ 0.75."),
        ("6. Scenario robustness.",
         "The optimal Nᵈᵃᴹ recommended by the median and conservative (P90) "
         "scenarios differs by fewer than 2 stories in 80% of all analyzed structures, "
         "demonstrating that a single median-optimized design provides near-optimal protection "
         "across the full range of seismic uncertainty captured by the eight-record ensemble. "
         "This reduces the need for separate optimization runs per hazard level in preliminary design."),
        ("Future work.",
         "Extensions of this framework should incorporate inelastic frame behavior, story-specific "
         "Mₑ distributions, installation cost constraints, and additional seismic records "
         "representative of different tectonic environments. Experimental validation of the "
         "NSD-based simulation results for multi-story configurations would further strengthen "
         "the confidence in the ML surrogate predictions. Integration of the trained models "
         "into an interactive design tool — already prototyped as a Streamlit application — "
         "represents an immediate path toward practical deployment."),
    ]

    for title_c, text_c in conclusions:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run_t = p.add_run(title_c + " ")
        run_t.bold = True
        run_t.font.size = Pt(11)
        run_t.font.name = 'Times New Roman'
        run_b = p.add_run(text_c)
        run_b.font.size = Pt(11)
        run_b.font.name = 'Times New Roman'
        doc.add_paragraph()

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════════
    # REFERENCIAS (clave)
    # ═══════════════════════════════════════════════════════════════════════════
    h1(doc, "REFERENCES (selected)")

    refs = [
        "Mualla, I.H. and Belev, B. (2002). Performance of steel frames with a new friction damper device under seismic excitation. Engineering Structures, 24(3), 365–371.",
        "Jarrahi, H., Asadi, A., Khatibinia, M. and Etedali, S. (2020). Optimal design of rotational friction dampers for improving seismic performance of inelastic structures. Journal of Building Engineering, 27, 100960.",
        "Jarrahi, H., Asadi, A., Khatibinia, M., Etedali, S. and Samadi, A. (2020). Simultaneous optimization of placement and parameters of rotational friction dampers for seismic-excited steel moment-resisting frames. Soil Dynamics and Earthquake Engineering, 136, 106193.",
        "Namorado, C. et al. Non-Smooth Dynamics formulation for rotational friction dampers. [Reference to be completed by authors].",
        "Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
        "McKay, M.D., Beckman, R.J. and Conover, W.J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics, 21(2), 239–245.",
        "Kazemi, F. et al. (2022). Machine learning-based prediction of seismic limit-state capacity of steel moment-resisting frames considering soil-structure interaction. Engineering Structures, 260, 114189.",
    ]

    for ref in refs:
        p = doc.add_paragraph(ref, style='List Number')
        p.runs[0].font.size = Pt(10)
        p.runs[0].font.name = 'Times New Roman'

    # ── Guardar ───────────────────────────────────────────────────────────────
    doc.save(OUTPUT_PATH)
    print(f"\n[OK] Paper generado: {OUTPUT_PATH.resolve()}")
    print(f"     Figuras incluidas desde: {FIGS_DIR.resolve()}")


if __name__ == "__main__":
    main()
