#!/usr/bin/env python3
"""Build 2026AGU_Vance_Electrical_rev1.pptx from 2026AGU_Vance_Electrical.pptx (original is opened
read-only and never written).  Re-runnable:

    python cortes2026/agu2026_poster/build_poster_rev1.py

Changes are documented in POSTER_REVISION.md.  Shapes are matched by NAME + TEXT, not by index.
"""
import copy
import os
import sys

from lxml import etree
from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(ROOT, '2026AGU_Vance_Electrical.pptx')
DST = os.path.join(ROOT, '2026AGU_Vance_Electrical_rev1.pptx')
FIG_A = os.path.join(HERE, 'fig_A_sigma_P.png')
FIG_B = os.path.join(HERE, 'fig_B_relative_sigma_0p9m.png')
FIG_C = os.path.join(HERE, 'fig_C_pct_increase_vs_T.png')
IN = 914400

# ---- text (numbers: see POSTER_REVISION.md for poster_facts.md sources)
HEAD1 = 'New: pressure raises conductivity more in colder oceans'
HEAD2 = ('NaCl 0.5 mol/kg to 700 MPa (0.8–35 °C); NaCl 0.9 mol/kg (1.7–29 °C) and '
         'KCl 0.70 mol/kg (6–30 °C) to 200 MPa')
METHOD = ('Method: two-electrode impedance spectroscopy (Gamry 600+) in the Simulator for Icy World Interiors. '
          'Each run’s 1-bar value is anchored to McCleskey (2011); the measured pressure dependence is kept. '
          'Compression and decompression agree within 0.34 percentage points (20–190 MPa; 0.9 mol/kg NaCl and '
          '0.70 mol/kg KCl, Nov 2026).')
PRELIM = ('Preliminary: the 0.5 mol/kg NaCl rise (13–22% by 200 MPa, Aug 2025) is about twice that at '
          '0.9 mol/kg and in Adams & Hall (1931; +6.6% at 25 °C). Cause under investigation.')
PRESS = [
    'New measurements (Aug 2025 – Nov 2026): conductivity rises with pressure to about 200 MPa. In 0.9 mol/kg '
    'NaCl the rise is 6% at 25–29 °C and 11% at 2–7 °C, equivalent to warming the ocean by '
    '3–4 °C. In 0.70 mol/kg KCl it is 7–13%.',
    'In 0.5 mol/kg NaCl, conductivity peaks near 200–250 MPa and then declines to 700 MPa, as in Adams & Hall (1931).',
    'The Pan et al. (2021) fit to data at 212–1713 MPa, extrapolated to lower pressure, gives +1% to −10% over '
    'the same range.',
    '→ Magnetic induction studies for Europa and Ganymede need careful assessment of the pressure effect on '
    'conductivity for relevant solutions',
]
FOOT = ('*Reference herein to any specific commercial product, process, or service by trade name, trademark, '
        'manufacturer, or otherwise, does not constitute or imply its endorsement by the United States Government or '
        'the Jet Propulsion Laboratory, California Institute of Technology.')

# ---- geometry (inches)
DIVIDER_END_Y = 31.6
FIGC_X, FIGC_Y, FIGC_W = 38.75, 33.25, 3.7         # height from aspect (7:6)
PRESS_X, PRESS_W, PRESS_BOTTOM = 38.75, 4.95, 42.0
PRESS_PT = 18


def find(shapes, name, text_start=None):
    hits = [s for s in shapes if s.name == name and (text_start is None or (s.has_text_frame and s.text_frame.text.strip().startswith(text_start)))]
    assert len(hits) == 1, (name, text_start, len(hits))
    return hits[0]


def set_xfrm(sh, x=None, y=None, w=None, h=None):
    if x is not None: sh.left = Emu(round(x * IN))
    if y is not None: sh.top = Emu(round(y * IN))
    if w is not None: sh.width = Emu(round(w * IN))
    if h is not None: sh.height = Emu(round(h * IN))


def png_size(p):
    from PIL import Image
    with Image.open(p) as im:
        return im.size


def new_box(slide, x, y, w, h, name):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    return tb


def add_par(tf, text, pt, bold=False, first=False, space_after=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    r = p.add_run()
    r.text = text
    r.font.size = Pt(pt)
    r.font.bold = bold          # no latin typeface set: inherits the theme font, like idx 84
    if space_after is not None:
        p.space_after = Pt(space_after)
    return p


def main():
    prs = Presentation(SRC)           # read-only: DST is a different path
    slide = prs.slides[0]
    shapes = list(slide.shapes)
    sw, sh_ = prs.slide_width / IN, prs.slide_height / IN

    # ---- A. new results band
    hd = new_box(slide, 0.5, 31.9, 25.2, 1.4, 'Rev1 band heading')
    add_par(hd.text_frame, HEAD1, 40, bold=True, first=True)
    add_par(hd.text_frame, HEAD2, 28)
    w, h = png_size(FIG_A); ha = 17.3 * h / w
    pa = slide.shapes.add_picture(FIG_A, Inches(0.5), Inches(33.6), Inches(17.3), Inches(ha)); pa.name = 'Rev1 Fig A sigma vs P'
    w, h = png_size(FIG_B); wb = 6.49 * w / h
    pb = slide.shapes.add_picture(FIG_B, Inches(18.0), Inches(33.6), Inches(wb), Inches(6.49)); pb.name = 'Rev1 Fig B relative sigma'
    assert 18.0 + wb < 25.9
    tx = new_box(slide, 0.5, 40.3, 25.3, 1.7, 'Rev1 band method text')
    add_par(tx.text_frame, METHOD, 20, first=True, space_after=4)
    add_par(tx.text_frame, PRELIM, 20)

    # divider: shorten to end at DIVIDER_END_Y
    div = find(shapes, 'Straight Connector 117')
    assert abs(div.left / IN - 25.96) < 0.05 and div.width == 0
    set_xfrm(div, h=DIVIDER_END_Y - div.top / IN)

    # ---- B. pressure section: replace picture, rewrite text
    old = find(shapes, 'Picture 150')
    assert abs(old.left / IN - 38.45) < 0.05 and abs(old.top / IN - 33.09) < 0.05
    w, h = png_size(FIG_C); hc = FIGC_W * h / w
    pc = slide.shapes.add_picture(FIG_C, Inches(FIGC_X), Inches(FIGC_Y), Inches(FIGC_W), Inches(hc))
    pc.name = 'Rev1 Fig C pct increase vs T'
    old._element.addprevious(pc._element)         # keep the old picture's z-order position
    old._element.getparent().remove(old._element)

    box = find(shapes, 'TextBox 151', 'New measurements in 5wt% NaCl')
    top = FIGC_Y + hc + 0.1
    set_xfrm(box, x=PRESS_X, y=top, w=PRESS_W, h=PRESS_BOTTOM - top)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    txBody = tf._txBody
    for p in list(txBody.findall(qn('a:p'))):
        txBody.remove(p)
    for i, t in enumerate(PRESS):
        p = etree.SubElement(txBody, qn('a:p'))
        if i < len(PRESS) - 1:
            ppr = etree.SubElement(p, qn('a:pPr'))
            sa = etree.SubElement(ppr, qn('a:spcAft')); etree.SubElement(sa, qn('a:spcPts')).set('val', '400')
        r = etree.SubElement(p, qn('a:r'))
        rpr = etree.SubElement(r, qn('a:rPr')); rpr.set('lang', 'en-US'); rpr.set('sz', str(PRESS_PT * 100)); rpr.set('dirty', '0')
        etree.SubElement(r, qn('a:t')).text = t

    # ---- C. unfinished items: image-source instruction -> footnote
    ft = find(shapes, 'TextBox 114', '*Thermo/Orion')
    p = ft.text_frame.paragraphs[0]._p
    runs = p.findall(qn('a:r'))
    runs[0].find(qn('a:t')).text = FOOT
    for r in runs[1:]:
        p.remove(r)
    for br in p.findall(qn('a:br')):
        p.remove(br)
    set_xfrm(ft, h=0.55)

    prs.save(DST)
    print('saved', DST, 'slide', sw, 'x', sh_, 'in; shapes', len(list(slide.shapes)))


if __name__ == '__main__':
    main()
