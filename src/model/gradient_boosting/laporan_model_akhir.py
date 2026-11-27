"""Laporan Markdown dan diagram reliabilitas SVG untuk tahap model akhir (RM-25 s.d. RM-27).

Port setia bagian laporan dari `Pemodelan PDH-PDL/model_akhir.py`. Semua fungsi hanya membaca
rekonsiliasi yang sudah dihitung oleh `model_akhir.run_model_akhir`; tidak ada perhitungan baru.
"""
import math

from pdhl_data.common import CLASSES

WARNA = {"sebelum": "#9aa0a6", "sesudah": "#1a73e8"}


# ---------------------------------------------------------------- gambar
def gambar_reliabilitas(rel_sebelum, rel_sesudah, n, tau):
    """Diagram reliabilitas SVG (tiga panel satu-lawan-semua), sebelum vs sesudah τ."""
    s, gap, kiri, atas = 200, 60, 56, 52
    lebar, tinggi = kiri + 3 * s + 2 * gap + 20, atas + s + 92
    el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{lebar}" height="{tinggi}" '
          f'viewBox="0 0 {lebar} {tinggi}" font-family="Helvetica, Arial, sans-serif" '
          'font-size="11">', f'<rect width="{lebar}" height="{tinggi}" fill="#ffffff"/>',
          f'<text x="{kiri}" y="22" font-size="14" font-weight="bold">Diagram reliabilitas '
          f'data kalibrasi 2024 (n = {n}), sebelum dan sesudah τ = {tau:.3f}</text>']
    for i, k in enumerate(CLASSES):
        x0, y0 = kiri + i * (s + gap), atas
        el += [f'<rect x="{x0}" y="{y0}" width="{s}" height="{s}" fill="none" stroke="#5f6368"/>',
               f'<line x1="{x0}" y1="{y0 + s}" x2="{x0 + s}" y2="{y0}" stroke="#bdc1c6" '
               'stroke-dasharray="4 3"/>',
               f'<text x="{x0 + s / 2}" y="{y0 - 8}" text-anchor="middle" '
               f'font-weight="bold">{k}</text>']
        for t in (0, 0.5, 1):
            el += [f'<text x="{x0 + t * s}" y="{y0 + s + 14}" text-anchor="middle">{t:g}</text>',
                   f'<text x="{x0 - 6}" y="{y0 + s - t * s + 4}" text-anchor="end">{t:g}</text>']
        for nama, rel in (("sebelum", rel_sebelum), ("sesudah", rel_sesudah)):
            titik = [(x0 + b["p_rerata"] * s, y0 + s - b["frek"] * s, b["n"])
                     for b in rel[k] if b["n"]]
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in titik)
            garis = ' stroke-dasharray="5 3"' if nama == "sebelum" else ""
            el.append(f'<polyline points="{pts}" fill="none" stroke="{WARNA[nama]}" '
                      f'stroke-width="1.8"{garis}/>')
            el += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2 + min(math.sqrt(m), 6) * 0.5:.1f}" '
                   f'fill="{WARNA[nama]}"><title>{nama}: n={m}</title></circle>'
                   for x, y, m in titik]
    yb = atas + s + 36
    el += [f'<text x="{kiri + 1.5 * s + gap}" y="{yb - 4}" text-anchor="middle">Peluang rata-rata '
           'per bin (10 bin selebar 0,1); sumbu tegak: frekuensi teramati</text>',
           f'<line x1="{kiri}" y1="{yb + 18}" x2="{kiri + 24}" y2="{yb + 18}" '
           f'stroke="{WARNA["sebelum"]}" stroke-width="1.8" stroke-dasharray="5 3"/>',
           f'<text x="{kiri + 30}" y="{yb + 22}">Sebelum τ (softmax skor mentah)</text>',
           f'<line x1="{kiri + 250}" y1="{yb + 18}" x2="{kiri + 274}" y2="{yb + 18}" '
           f'stroke="{WARNA["sesudah"]}" stroke-width="1.8"/>',
           f'<text x="{kiri + 280}" y="{yb + 22}">Sesudah τ</text>',
           f'<text x="{kiri + 400}" y="{yb + 22}" fill="#5f6368">Garis putus abu-abu: kalibrasi '
           'sempurna; ukuran titik ∝ √n</text>', "</svg>"]
    return "\n".join(el) + "\n"


# ---------------------------------------------------------------- laporan
def _f(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def _r(x):
    return "–" if x is None else f"{x:+.4f}"


def _baris_metrik(nama, s):
    return (f"| {nama} | {s['n']} | {_f(s['log_loss'])} | {_f(s['brier'])} | "
            + " | ".join(_f(s["ece"][k]) for k in CLASSES) + " |")


def _tabel_strategi(pr):
    lines = ["| Strategi | Cakupan | Rev | Cont | Abstain | R/event | R/rekomendasi |",
             "|---|---|---|---|---|---|---|"]
    baris = [(f"SM δ = {h['delta']:g}", h) for h in pr["SM"]]
    baris += [(f"{nm} ({ket})", pr[nm]) for nm, ket in (("S0", "semua reversal"),
                                                        ("S1", "reversal bila wick ≥ 2/3"))]
    for nama, h in baris:
        lines.append(f"| {nama} | {h['cakupan']:.1%} | {h['n_rev']} | {h['n_cont']} | "
                     f"{h['n_abstain']} | {h['r_per_event']:+.4f} | "
                     f"{_r(h['r_per_rekomendasi'])} |")
    return lines


def _bagian_model(recon):
    cfg, env = recon["konfigurasi"], recon["lingkungan"]
    return [
        "# Laporan Model Akhir dan Kalibrasi τ (RM-25)", "",
        "Sumber: `model.gradient_boosting.model_akhir`. Lingkungan: Python "
        f"{env['python']}, LightGBM {env['lightgbm']}, NumPy {env['numpy']} ({env['platform']}). "
        "Data uji tidak dimuat.", "",
        "## 1. Model akhir", "",
        f"- Konfigurasi terpilih walk-forward: **{cfg['id']}** (max_depth {cfg['max_depth']}, "
        f"n_estimators {cfg['n_estimators']}, min_child_samples {cfg['min_child_samples']}); "
        "parameter lain sesuai Kunci-1.",
        f"- Dilatih sekali pada seluruh dev 2016–2023: {recon['n_latih']} event, purging di batas "
        f"2024-01-01: {recon['n_purged']} event. Jumlah pohon {recon['jumlah_pohon']} "
        f"({cfg['n_estimators']} iterasi × 3 kelas).",
        f"- Berkas model `{recon['model_berkas']}`, SHA-256 `{recon['model_sha256']}`.",
        f"- Log loss dev dalam-sampel {_f(recon['dev_dalam_sampel']['log_loss'])} vs log loss OOF "
        f"walk-forward {_f(recon['dev_dalam_sampel']['oof_walk_forward'])} (dalam-sampel selalu "
        "lebih optimistis; hanya sebagai keterangan).", ""]


def _bagian_kalibrasi(recon):
    k, pc = recon["kalibrasi"], recon["pembanding_calib"]
    arah = ("lebih dari 1: model terlalu yakin, peluang dilunakkan" if k["tau"] > 1 else
            "kurang dari 1: model kurang yakin, peluang dipertajam")
    lines = [
        "## 2. Kalibrasi temperature scaling (data 2024)", "",
        f"τ = **{k['tau']:.4f}** ({arah}). Dicari dengan golden-section pada "
        f"[{k['rentang'][0]:g}; {k['rentang'][1]:g}] ({k['iterasi']} iterasi, toleransi "
        f"{k['toleransi']:g}); grid log-spasi {k['cek_grid']['n']} titik memberi τ "
        f"{k['cek_grid']['tau']:.4f} dengan log loss {_f(k['cek_grid']['log_loss'], 6)}, jadi "
        "minimumnya global." + (" **τ berada di batas rentang.**" if k["di_batas"] else ""), "",
        f"| Prediktor (2024) | n | Log loss | Brier | ECE {CLASSES[0]} | ECE {CLASSES[1]} | "
        f"ECE {CLASSES[2]} |", "|---|---|---|---|---|---|---|",
        _baris_metrik("Model sebelum τ", k["sebelum"]),
        _baris_metrik("**Model sesudah τ**", k["sesudah"]),
        _baris_metrik("B0 (base rate dev)", pc["B0"]),
        _baris_metrik("B1 (wick ≥ 2/3, dev)", pc["B1"]), "",
        "Catatan: model sebelum τ, B0, dan B1 tidak pernah melihat data 2024. Model sesudah τ "
        "memakai 2024 untuk mengestimasi satu parameter (τ), jadi angkanya sedikit optimistis. "
        "Perbandingan resmi model vs B0/B1 hanya pada data uji (RQ1). "
        "Gambar: `gambar_reliabilitas_2024.svg`.", "",
        "### Log loss per jenis level (2024)", "",
        "| Level | n | Model sebelum τ | Model sesudah τ | B0 | B1 |", "|---|---|---|---|---|---|"]
    for lv, s in recon["per_level_calib"].items():
        lines.append(f"| {lv} | {s['n']} | {_f(s['model_sebelum'])} | {_f(s['model_sesudah'])} | "
                     f"{_f(s['B0'])} | {_f(s['B1'])} |")
    lines += ["", "### Tabel reliabilitas (bin berisi saja)", "",
              "| Kelas | Bin | n | p̄ sebelum | Frek. | n | p̄ sesudah | Frek. |",
              "|---|---|---|---|---|---|---|---|"]
    for kelas in CLASSES:
        for b0, b1 in zip(k["sebelum"]["reliabilitas"][kelas], k["sesudah"]["reliabilitas"][kelas]):
            if b0["n"] or b1["n"]:
                lines.append(f"| {kelas} | {b0['batas'][0]:.1f}–{b0['batas'][1]:.1f} | {b0['n']} | "
                             f"{_f(b0['p_rerata'], 3)} | {_f(b0['frek'], 3)} | {b1['n']} | "
                             f"{_f(b1['p_rerata'], 3)} | {_f(b1['frek'], 3)} |")
    return lines


def _bagian_keputusan(recon):
    m, pr, oof = recon["aturan_keputusan"], recon["pratinjau_calib"], recon["cakupan_oof_dev"]
    lines = ["", "## 3. Parameter aturan keputusan", "",
             f"- `m_rev` = **{m['m_rev']:+.4f}R**, `m_cont` = **{m['m_cont']:+.4f}R** (rata-rata "
             f"R pada {m['n_netral']} event Netral dev; `m_cont` = −`m_rev` · b/a).",
             f"- δ = {m['delta']}R (kunci); grid kurva cakupan {m['delta_grid']}.",
             f"- E(rev) = p_rev · {m['a'] / m['b']:g} − p_cont + p_net · m_rev; E(cont) = p_cont · "
             f"{m['b'] / m['a']:g} − p_rev + p_net · m_cont. {m['aturan_seri']}", "",
             "## 4. Pratinjau aturan keputusan (deskriptif, R kotor tanpa biaya)", "",
             f"### 4.1 Data kalibrasi 2024 (n = {pr['S0']['n']})", "", pr["catatan"], ""]
    lines += _tabel_strategi(pr)
    lines += ["", f"### 4.2 OOF walk-forward dev 2019–2023 (n = {oof['S0']['n']})", "",
              oof["catatan"], ""]
    lines += _tabel_strategi(oof)
    lines += ["", "## 5. Komponen calon Kunci-2", "",
              "Model (berkas + SHA-256), konfigurasi, τ, `m_rev`, `m_cont`, δ, peluang B0/B1 dev, "
              "kolom fitur, dan versi pustaka tercatat di `model_akhir_rekonsiliasi.json`. "
              "Templat SHAP dan penanda commit ditambahkan saat Kunci-2.", ""]
    return lines


def laporan(recon):
    return "\n".join(_bagian_model(recon) + _bagian_kalibrasi(recon) + _bagian_keputusan(recon))
