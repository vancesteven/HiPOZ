#!/usr/bin/env python3
"""Build 2026AGU_Vance_Electrical_rev2.pptx from the ORIGINAL draft 2026AGU_Vance_Electrical.pptx
(opened read-only, never written).  Re-runnable end to end:

    python cortes2026/agu2026_poster/build_poster_rev2.py [--title "Official AGU title"] [--no-preview]

Shapes of the draft are matched by NAME + TEXT (+ position check), never by index alone.
Numbers on the poster come only from poster_facts.md; see POSTER_REVISION_rev2.md.
"""
import argparse
import copy
import hashlib
import json
import os
import sys

from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------- constants
TITLE = "[PASTE AGU TITLE HERE]"          # the author pastes the official AGU title here (or --title)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(ROOT, '2026AGU_Vance_Electrical.pptx')
DST = os.path.join(ROOT, '2026AGU_Vance_Electrical_rev2.pptx')
FIG_A = os.path.join(HERE, 'fig_A_sigma_P.png')
FIG_B = os.path.join(HERE, 'fig_B_relative_sigma_0p9m.png')
FIG_C = os.path.join(HERE, 'fig_C_pct_increase_vs_T.png')
RECON = os.environ.get(
    'POSTER_RECON',
    '/tmp/claude-1001/-Users-svance-src-hipozgenai/19b43f9f-e4ff-460d-9e45-e04fb42202fa/scratchpad/poster_recon')
IN = 914400
PT_PER_LINE = 1.2                 # line height = 1.2 x font size (PowerPoint single spacing, approx.)
MAX_USE = 0.90                    # text-fit failure threshold

AUTHORS = [('Steven D. Vance', '1'), ('Jesus Cortes', '2'), ('Rose Garrity', '3'), ('Zachary Diaz', '4'),
           ('Jessica M. Weber', '1'), ('Alicja Urbanska', '5'), ('Leïla Mahboub', '6'),
           ('Kristjan Stone', '1'), ('Abby Kavner', '3')]
AFFILS = [('1', 'Jet Propulsion Laboratory, California Institute of Technology, Pasadena;  '),
          ('2', 'Colby College, Waterville, Maine; '),
          ('3', 'University of California, Los Angeles; '),
          ('4', 'University of Nevada, Las Vegas; '),
          ('5', 'Minerva University, San Francisco; '),
          ('6', 'École Supérieure de Physique et de Chimie Industrielles de la Ville de Paris.  '
                'steven.d.vance@jpl.nasa.gov')]

HEAD_MOTIV = 'Why ocean conductivity matters'
BODY_MOTIV = ('Joint inversion of gravity and magnetic induction measurements can constrain the presence, thickness, '
              'and salinity of oceans at Jupiter’s large icy moons (Vance et al. 2023; Roberts et al. 2023; '
              'Van Hoolst et al. 2024). Interpreting induction signals needs electrical conductivity data for aqueous '
              'solutions at ocean pressures and temperatures. We are filling gaps with lab measurements in binary and '
              'mixed solutions (Mahboub et al. 2026).')
HEAD_HERO = 'Pressure raises conductivity more in colder oceans'
TAKEAWAY = [
    'By 200 MPa, the pressure at Europa’s seafloor, conductivity of 0.9 mol/kg NaCl is up 6% at 25–29 °C and '
    '11% at 2–7 °C.',
    'That equals warming the ocean by 3–4 °C.',
    'The Pan et al. (2021) fit to data at 212–1713 MPa, extrapolated down, gives +1% to −10% instead.',
]
HEAD_NEW = 'New high-pressure measurements (Aug 2025 – Nov 2026)'
SUB_NEW = ('NaCl 0.5 mol/kg to 700 MPa (0.8–35 °C); NaCl 0.9 mol/kg (1.7–29 °C) and KCl 0.70 mol/kg (6–30 °C) '
           'to 200 MPa')
CAP_C = 'Rise in conductivity from 1 bar to 200 MPa'
FIND_HEAD = 'What we find'
FINDINGS = [
    'In 0.70 mol/kg KCl the rise is 7–13%, on average about 2 percentage points above NaCl at the same temperature '
    '(0.9–2.6).',
    'In 0.5 mol/kg NaCl, conductivity peaks near 200–250 MPa and then declines to 700 MPa, as in Adams & Hall (1931).',
    'Preliminary: the 0.5 mol/kg rise (13–22% by 200 MPa) is about twice that at 0.9 mol/kg and in Adams & Hall '
    '(+6.6% at 25 °C). Cause under investigation.',
    '→ Magnetic induction studies for Europa and Ganymede need careful assessment of the pressure effect on '
    'conductivity for relevant solutions.',
]
METH_HEAD = 'Method'
METHOD = ('Two-electrode impedance spectroscopy (Gamry 600+) in the Simulator for Icy World Interiors, a hydraulic '
          'high-pressure system. Each run’s 1-bar value is anchored to McCleskey (2011); the measured pressure '
          'dependence is kept. Compression and decompression agree within 0.34 percentage points (20–190 MPa; '
          '0.9 mol/kg NaCl and 0.70 mol/kg KCl, Nov 2026).')
FOOT = ('*Reference herein to any specific commercial product, process, or service by trade name, trademark, '
        'manufacturer, or otherwise, does not constitute or imply its endorsement by the United States Government or '
        'the Jet Propulsion Laboratory, California Institute of Technology.')
HEAD_EARLY = 'How Does Pressure Affect Electrical Conductivity?'

# references: (bold lead, regular text, italic journal/rest) -> list of (text, bold, italic) runs, alphabetical
REFS = [
    [('Adams, ', 1, 0), ('L. H., Hall, R. E. (1931). ', 0, 0), ('J. Phys. Chem', 0, 1),
     (', 35(8):2145–2163, 10.1021/j150326a001. ', 0, 0)],
    [('Guo H., ', 1, 0), ('Keppler, H. (2019) ', 0, 0), ('JGR: Solid Earth', 0, 1),
     (', 124(2):1397–1411, 10.1029/2018jb016658. ', 0, 0)],
    [('Defense Lab: Kraichman', 1, 0), (', M. B. (1964) ', 0, 0),
     ('Defense Doc. Center Sci. Tech. Info.', 0, 1), (' ', 0, 0)],
    [('Larionov, ', 1, 0), ('E. G., Kryukov, P. A. (1984) ', 0, 0),
     ('Izvestiya Sibirskogo Otdeleniya Akademii Nauk SSSR Seriya Khimicheskikh Nauk', 0, 1), (', (5):20–23. ', 0, 0)],
    [('Mahboub, ', 1, 0), ('L., Weber, J., Cortes, J., Psarakis, C., Melwani Daswani, M., Vance, S. D. (2026). '
                           'Electrical properties of icy world oceans from laboratory measurements. ', 0, 0),
     ('ACS Earth Space Chem', 0, 1), (', 10.1021/acsearthspacechem.5c00333. ', 0, 0)],
    [('McCleskey ', 1, 0), (' (2011) ', 0, 0), ('J. Chem Eng. Data', 0, 1), (', 56(2):317–327. ', 0, 0)],
    [('McCleskey, ', 1, 0), ('R. B., Nordstrom, D. K., Ryan, J. N., Ball, J. W. (2012). A new method of calculating '
                             'electrical conductivity with applications to natural waters. ', 0, 0),
     ('Geochim. Cosmochim. Acta', 0, 1), (', 77:369–382. ', 0, 0)],
    [('McDougall ', 1, 0), ('& Barker (2011, Gibbs Seawater) [full reference to add]. ', 0, 0)],
    [('Pan ', 1, 0), ('et al. (2021) ', 0, 0), ('GRL', 0, 1), (', 48(17), 10.1029/2021GL094020 ', 0, 0)],
    [('Psarakis ', 1, 0), ('et al. (2024) ', 0, 0), ('ACS Earth Space Chem.', 0, 1),
     (', 8(6):1146–1153, 10.1021/acsearthspacechem.3c00345. ', 0, 0)],
    [('Roberts, ', 1, 0), ('J. H. et al. (2023). Exploring the interior of Europa with Europa Clipper. ', 0, 0),
     ('Space Sci. Rev.', 0, 1), (', 219(6), 46. ', 0, 0)],
    [('Sinmyo ', 1, 0), ('R., Keppler, H. (2016) ', 0, 0), ('Contributions to Mineralogy and Petrology', 0, 1),
     (', 172(1), 10.1007/s00410-016-1323-z. ', 0, 0)],
    [('Vance, ', 1, 0), ('S. D. et al. (2023). Investigating Europa’s habitability with Europa Clipper. ', 0, 0),
     ('Space Sci. Rev.', 0, 1), (', 219(8), 81. ', 0, 0)],
    [('Van Hoolst, ', 1, 0), ('T. et al. (2024). Geophysical characterization of the interiors of Ganymede, Callisto '
                              'and Europa by ESA’s JUpiter Icy moons Explorer. ', 0, 0),
     ('Space Sci. Rev.', 0, 1), (', 220(5), 54. ', 0, 0)],
]

# --------------------------------------------------------------------------- text measurement
_MPL = None


def _font_paths():
    import matplotlib
    d = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data/fonts/ttf')
    return os.path.join(d, 'DejaVuSans.ttf'), os.path.join(d, 'DejaVuSans-Bold.ttf')


_FC = {}


def pil_font(pt, bold=False, scale=10):
    """DejaVu Sans (substitute for the theme font Aptos; ~10-15% wider, hence conservative)."""
    reg, bld = _font_paths()
    k = (round(pt * scale), bold)
    if k not in _FC:
        _FC[k] = ImageFont.truetype(bld if bold else reg, max(1, round(pt * scale)))
    return _FC[k]


def wrap_lines(text, pt, bold, width_pt):
    """Greedy word wrap; returns list of lines. Width in points."""
    f = pil_font(pt, bold)
    lines = []
    for hard in text.split('\n'):
        words = hard.split(' ')
        cur = ''
        for w in words:
            t = (cur + ' ' + w) if cur else w
            if f.getlength(t) / 10.0 <= width_pt or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def spec_text(p):
    return ''.join(r[0] for r in p['runs'])


def spec_size(p):
    return max(r[1].get('sz', 18) for r in p['runs'])


def spec_bold(p):
    n = sum(len(r[0]) for r in p['runs'] if r[1].get('b'))
    return n * 2 > sum(len(r[0]) for r in p['runs'])


def measure(paras, w_in, inset=(0.1, 0.05, 0.1, 0.05)):
    """Estimated text height in inches for the paragraph specs in a box w_in wide."""
    width_pt = (w_in - inset[0] - inset[2]) * 72.0
    tot = 0.0
    for p in paras:
        ind = p.get('marL', 0.0)
        sz = spec_size(p)
        lines = wrap_lines(spec_text(p), sz, spec_bold(p), width_pt - ind * 72.0)
        tot += len(lines) * sz * PT_PER_LINE + p.get('before', 0) + p.get('after', 0)
    return tot / 72.0 + inset[1] + inset[3]


FITS = []   # (name, est_h, box_h, minimum pt, smallest pt used)


def register_fit(name, paras, w, h, inset=(0.1, 0.05, 0.1, 0.05)):
    est = measure(paras, w, inset)
    FITS.append((name, est, h, min(spec_size(p) for p in paras)))
    return est


# --------------------------------------------------------------------------- XML helpers
def emu(v):
    return int(round(v * IN))


def mk_rpr(tag, sz=None, b=None, i=None, sup=None, color=None, u=None):
    r = etree.Element(qn(tag))
    r.set('lang', 'en-US')
    if sz is not None:
        r.set('sz', str(int(round(sz * 100))))
    if b is not None:
        r.set('b', '1' if b else '0')
    if i:
        r.set('i', '1')
    if sup:
        r.set('baseline', '30000')
    r.set('dirty', '0')
    if color:
        sf = etree.SubElement(r, qn('a:solidFill'))
        etree.SubElement(sf, qn('a:srgbClr')).set('val', color)
    return r        # no <a:latin>: inherits the theme font (Aptos), like the draft's headings


def mk_p(spec):
    p = etree.Element(qn('a:p'))
    ppr = etree.SubElement(p, qn('a:pPr'))
    if spec.get('bullet'):
        ml = emu(spec.get('marL', 0.4))
        ppr.set('marL', str(ml))
        ppr.set('indent', str(-ml))
    else:
        ppr.set('marL', '0')
        ppr.set('indent', '0')
    ppr.set('algn', spec.get('algn', 'l'))
    ln = etree.SubElement(ppr, qn('a:lnSpc'))
    etree.SubElement(ln, qn('a:spcPct')).set('val', '100000')
    sb = etree.SubElement(ppr, qn('a:spcBef'))
    etree.SubElement(sb, qn('a:spcPts')).set('val', str(int(spec.get('before', 0) * 100)))
    sa = etree.SubElement(ppr, qn('a:spcAft'))
    etree.SubElement(sa, qn('a:spcPts')).set('val', str(int(spec.get('after', 0) * 100)))
    if spec.get('bullet'):
        etree.SubElement(ppr, qn('a:buFont')).set('typeface', 'Arial')
        etree.SubElement(ppr, qn('a:buChar')).set('char', spec['bullet'])
    else:
        etree.SubElement(ppr, qn('a:buNone'))
    for text, st in spec['runs']:
        r = etree.SubElement(p, qn('a:r'))
        r.append(mk_rpr('a:rPr', st.get('sz'), st.get('b'), st.get('i'), st.get('sup'), st.get('color'), st.get('u')))
        etree.SubElement(r, qn('a:t')).text = text
    return p


def P(text, sz, b=False, **kw):
    d = {'runs': [(text, {'sz': sz, 'b': b})]}
    d.update(kw)
    return d


def set_body(shape, paras, wrap=True, inset=None, anchor='t'):
    """Replace all paragraphs of an existing/new text shape; fixed size (no autofit)."""
    tx = shape._element.txBody
    for p in tx.findall(qn('a:p')):
        tx.remove(p)
    for sp in paras:
        tx.append(mk_p(sp))
    bp = tx.find(qn('a:bodyPr'))
    for ch in list(bp):
        bp.remove(ch)
    bp.set('wrap', 'square' if wrap else 'none')
    bp.set('anchor', anchor)
    if inset:
        for k, v in zip(('lIns', 'tIns', 'rIns', 'bIns'), inset):
            bp.set(k, str(emu(v)))
    etree.SubElement(bp, qn('a:noAutofit'))


def new_box(slide, x, y, w, h, name, paras, fill=None, line=None, inset=(0.1, 0.05, 0.1, 0.05), anchor='t'):
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    set_body(s, paras, True, inset, anchor)
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line:
        s.line.color.rgb = RGBColor.from_string(line)
        s.line.width = Pt(1.5)
    register_fit(name, paras, w, h, inset)
    return s


def set_xywh(sh, x=None, y=None, w=None, h=None):
    if x is not None:
        sh.left = Emu(emu(x))
    if y is not None:
        sh.top = Emu(emu(y))
    if w is not None:
        sh.width = Emu(emu(w))
    if h is not None:
        sh.height = Emu(emu(h))


def geom(sh):
    return [v / IN for v in (sh.left, sh.top, sh.width, sh.height)]


def find(shapes, name, text_start=None, near=None, tol=0.15):
    hits = []
    for s in shapes:
        if s.name != name:
            continue
        if text_start is not None and not (s.has_text_frame and s.text_frame.text.strip().startswith(text_start)):
            continue
        if near is not None:
            gx, gy = s.left / IN, s.top / IN
            if abs(gx - near[0]) > tol or abs(gy - near[1]) > tol:
                continue
        hits.append(s)
    assert len(hits) == 1, (name, text_start, near, len(hits))
    return hits[0]


def delete(sh):
    sh._element.getparent().remove(sh._element)


def theme_font_all(shape):
    """Point every run of the shape at the theme font (keeps sizes; leaves code-font runs alone)."""
    n = 0
    for rpr in shape._element.iter(qn('a:rPr'), qn('a:endParaRPr'), qn('a:defRPr')):
        for tag in ('a:latin', 'a:cs'):
            el = rpr.find(qn(tag))
            if el is not None and el.get('typeface') not in ('+mn-lt', '+mn-cs', 'Courier New'):
                rpr.remove(el)
                n += 1
    return n


def add_line(slide, x1, y1, x2, y2, name):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.name = name
    c.line.color.rgb = RGBColor(0x9A, 0x9A, 0x9A)
    c.line.width = Pt(2.5)
    return c


# --------------------------------------------------------------------------- build
def choose_title_size(title, width_in):
    width_pt = (width_in - 0.2) * 72.0
    for sz in range(88, 65, -2):
        if len(wrap_lines(title, sz, True, width_pt)) <= 2:
            return sz, 2
    return 66, len(wrap_lines(title, 66, True, width_pt))


def build(title):
    prs = Presentation(SRC)            # read-only: DST is a different path
    slide = prs.slides[0]
    S = list(slide.shapes)
    notes = []

    # ---------------------------------------------------------------- 1. deletions (name + text + position)
    DEL = [
        ('Picture 150', None, (38.45, 33.09)),                       # old this-work sigma(P) plot
        ('Title 1', None, (14.5, 13.68)), ('AutoShape 10', None, (16.45, 12.68)),
        ('Picture 28', None, (19.94, 11.5)),                          # Europa cutaway
        ('TextBox 29', 'Chemical', None), ('TextBox 30', 'Redox', None), ('TextBox 31', 'pH', None),
        ('TextBox 32', 'Adiabatic', None), ('Straight Arrow Connector 33', None, None),
        ('TextBox 34', 'Material', None), ('TextBox 35', 'Image: NASA', None),
        ('Straight Arrow Connector 39', None, None),
        ('Title 1', 'Europa’s Interior', None), ('Title 1', 'Are Dictated', None),
        ('TextBox 43', 'Redox', None), ('Straight Arrow Connector 44', None, None),
        ('TextBox 45', 'ice thickness', None), ('TextBox 46', 'ice thickness', None),
        ('Picture 47', None, None), ('TextBox 48', '28% increase', None), ('Picture 49', None, None),
        ('TextBox 50', 'Up to 10.8', None), ('TextBox 51', 'ocean density', None),
        ('TextBox 52', 'ocean conductivity', None), ('TextBox 53', '(Chang', None),
        ('TextBox 54', 'Ice-Water', None), ('TextBox 55', None, (17.52, 12.0)), ('Title 1', None, (0.49, 12.07)),
        ('TextBox 70', None, (3.24, 14.26)), ('Group 171', None, None),
        ('Title 1', 'Can Europa Clipper', None), ('Picture 81', None, None),
        ('Straight Arrow Connector 90', None, None),
        ('Rectangle 107', None, None), ('TextBox 109', '*', None),
        ('Straight Connector 117', None, None), ('Straight Connector 118', None, None),
        ('Straight Connector 122', None, None),
        ('Right Arrow 142', None, None), ('TextBox 151', 'New measurements in 5wt%', None),
        ('TextBox 165', 'This can run', None), ('TextBox 183', 'Vance et al. (2018', None),
        ('Picture 188', None, (8.48, 4.61)),
    ]
    removed = []
    for name, txt, near in DEL:
        sh = find(S, name, txt, near, tol=0.3)
        removed.append((name, (sh.text_frame.text[:50] if sh.has_text_frame else '')))
        delete(sh)
    S = list(slide.shapes)
    # blank rectangle inside the compilation group (Group 1 > Group 131 > Rectangle 133)
    g1 = find(S, 'Group 1')
    g131 = [c for c in g1.shapes if c.name == 'Group 131'][0]
    rect = [c for c in g131.shapes if c.name == 'Rectangle 133'][0]
    delete(rect)
    removed.append(('Rectangle 133 (in Group 131)', ''))

    # ---------------------------------------------------------------- 2. header
    tsz, tlines = choose_title_size(title, 31.0)
    title_h = max(2, tlines) * tsz * PT_PER_LINE / 72.0 + 0.15
    t = find(S, 'TextBox 83')
    set_xywh(t, 0.5, 0.5, 31.0, title_h)
    set_body(t, [P(title, tsz, True)])
    register_fit('Title (TITLE)', [P(title, tsz, True)], 31.0, title_h)
    if tlines > 2:
        notes.append('WARNING: title needs %d lines even at 66 pt' % tlines)

    def author_paras():
        runs = []
        for i, (n, a) in enumerate(AUTHORS):
            runs.append(((', ' if i else '') + n, {'sz': 28, 'b': True}))
            runs.append((a, {'sz': 28, 'b': True, 'sup': True}))
        return [{'runs': runs}]

    ay = 0.5 + title_h
    ap = author_paras()
    ah = measure(ap, 31.0) / 0.85
    a = find(S, 'TextBox 85')
    set_xywh(a, 0.5, ay, 31.0, ah)
    set_body(a, ap)
    register_fit('Authors', ap, 31.0, ah)
    fruns = []
    for num, txt in AFFILS:
        fruns.append((num, {'sz': 18, 'sup': True}))
        fruns.append((txt, {'sz': 18}))
    fp = [{'runs': fruns}]
    fh = measure(fp, 31.0) / 0.85
    f = find(S, 'TextBox 87')
    set_xywh(f, 0.5, ay + ah, 31.0, fh)
    set_body(f, fp)
    register_fit('Affiliations', fp, 31.0, fh)
    header_bottom = ay + ah + fh
    logo = find(S, 'Picture 127')
    set_xywh(logo, x=43.5 - logo.width / IN, y=0.9)

    # ---------------------------------------------------------------- 3. Row 1
    Y1 = max(5.6, header_bottom + 0.3)
    notes.append('header bottom y = %.2f, Row 1 top = %.2f' % (header_bottom, Y1))

    # left column: motivation
    LX, LW = 0.5, 13.0
    h1 = new_box(slide, LX, Y1, LW, 0.95, 'Rev2 heading motivation', [P(HEAD_MOTIV, 40, True)], inset=(0.1, 0.05, 0.1, 0.05))
    bp = [P(BODY_MOTIV, 24)]
    bh = measure(bp, LW) / 0.85
    body = new_box(slide, LX, Y1 + 1.0, LW, bh, 'Rev2 body motivation', bp)
    ind_fig = find(S, 'Picture 92')
    ind_txt = find(S, 'Content Placeholder 2')
    # induction explanatory text: keep wording, 24 pt, left aligned, tight spacing
    ip = [P(r.text, 24, before=(0 if i == 0 else 4)) for i, r in enumerate([pp.runs[0] for pp in ind_txt.text_frame.paragraphs])]
    ih = measure(ip, LW) / 0.85
    fig_top = Y1 + 1.0 + bh + 0.15
    fig_w = LW
    fig_h = fig_w * ind_fig.height / ind_fig.width
    # shrink the figure if the column would otherwise run past y 21.9
    while fig_top + fig_h + 0.1 + ih > 21.9 and fig_w > 10.0:
        fig_w -= 0.1
        fig_h = fig_w * ind_fig.height / ind_fig.width
    set_xywh(ind_fig, LX + (LW - fig_w) / 2, fig_top, fig_w, fig_h)
    set_xywh(ind_txt, LX, fig_top + fig_h + 0.1, LW, ih)
    set_body(ind_txt, ip)
    register_fit('Induction explanatory text', ip, LW, ih)
    left_bottom = fig_top + fig_h + 0.1 + ih
    notes.append('induction figure width %.2f in (column %.1f)' % (fig_w, LW))

    # hero column
    HX, HW = 14.0, 12.0
    hh = measure([P(HEAD_HERO, 40, True)], 12.0) / 0.85
    new_box(slide, HX, Y1, HW, hh, 'Rev2 heading hero', [P(HEAD_HERO, 40, True)])
    pb = slide.shapes.add_picture(FIG_B, Inches(HX), Inches(Y1 + hh + 0.1), Inches(12.0), Inches(10.0))
    pb.name = 'Rev2 Fig B relative sigma'
    tk = [P(t_, 26, after=(8 if i < 2 else 0)) for i, t_ in enumerate(TAKEAWAY)]
    tk_y = Y1 + hh + 0.1 + 10.0 + 0.2
    tk_h = measure(tk, HW, (0.2, 0.15, 0.2, 0.15)) / 0.85
    new_box(slide, HX, tk_y, HW, tk_h, 'Rev2 takeaway box', tk, fill='EAF1F8', line='5B7FA6', inset=(0.2, 0.15, 0.2, 0.15))
    hero_bottom = tk_y + tk_h
    row1_bottom = max(left_bottom, hero_bottom)

    # ---------------------------------------------------------------- 4. benchtop block (right column)
    BX_R = 43.6
    heading = find(S, 'TextBox 94')
    rest_names = [('Picture 130', None), ('TextBox 93', None), ('Picture 97', None), ('Picture 96', None),
                  ('Group 99', None), ('TextBox 98', None), ('TextBox 104', None), ('TextBox 105', None),
                  ('TextBox 106', None), ('Picture 108', None), ('Picture 110', None), ('Picture 111', None),
                  ('TextBox 112', None), ('TextBox 113', None)]
    rest = [find(S, n, t) for n, t in rest_names]
    # --- small non-rigid fixes inside the block (draft coordinates, inches): the draft's stacked text boxes
    # overlapped each other and the circuit image; box heights are cut to the text and the stack is nudged down.
    by = {sh.name: sh for sh in rest}
    set_xywh(by['TextBox 104'], y=18.32, h=0.42)
    set_xywh(by['TextBox 106'], y=18.74, h=1.55)
    set_xywh(by['TextBox 98'], y=20.29, h=3.05)
    set_xywh(by['Picture 108'], x=32.12, y=23.36)          # clear of the melting-line diagram (x 34.36)
    set_xywh(by['TextBox 105'], y=24.34, h=0.42)
    ph0 = by['Picture 97'].height
    by['Picture 97'].crop_bottom = 0.057                    # photo's blank bottom strip (30 of 538 px)
    by['Picture 97'].height = Emu(int(round(ph0 * (1 - 0.057))))
    right = max(sh.left / IN + sh.width / IN for sh in rest)
    dx = BX_R - right
    top_rest = min(sh.top / IN for sh in rest)
    RC_X = 26.48
    RC_W = BX_R - RC_X
    hp = [{'runs': [('Laboratory investigation of electrical conductivity: ', {'sz': 32, 'b': True}),
                    ('identifying where more precise lab data are needed', {'sz': 32})]}]
    hh2 = measure(hp, RC_W) / 0.85
    hd_bottom = Y1 + hh2
    dy = (hd_bottom + 0.2) - top_rest
    for sh in rest:
        sh.left = Emu(sh.left + emu(dx))
        sh.top = Emu(sh.top + emu(dy))
    # heading: 32 pt as in the draft, first run bold, trailing empty paragraph dropped
    set_xywh(heading, RC_X, Y1, RC_W, hh2)
    set_body(heading, hp)
    register_fit('Benchtop heading', hp, RC_W, hh2)
    photo = find(S, 'Picture 97')
    # crop the blank strip at the bottom of the meter photo (35 of 538 px are white) so that the
    # "Benchtop conductivity sensor:" line below it no longer overlaps the picture
    px, py, pw, ph = geom(photo)
    # asterisk next to the photo and the disclaimer footnote beside it
    ast = new_box(slide, px - 0.42, py + ph - 1.0, 0.4, 0.6, 'Rev2 photo asterisk', [P('*', 32)], inset=(0.02, 0.0, 0.02, 0.0))
    fp2 = [{'runs': [(FOOT, {'sz': 14, 'color': '32373B'})]}]
    fw = (px - 0.45) - RC_X
    fh2 = measure(fp2, fw) / 0.85
    foot = find(S, 'TextBox 114')
    set_xywh(foot, RC_X, py + ph - fh2 - 0.05, fw, fh2)
    set_body(foot, fp2)
    register_fit('Disclaimer footnote', fp2, fw, fh2)
    # fonts: normalise to theme font in the kept benchtop boxes
    nfont = 0
    for n, tx in [('TextBox 93', None), ('TextBox 98', None), ('TextBox 104', None), ('TextBox 105', None),
                  ('TextBox 106', None), ('TextBox 112', None), ('TextBox 113', None), ('TextBox 114', None)]:
        nfont += theme_font_all(find(list(slide.shapes), n, tx))
    g99 = find(list(slide.shapes), 'Group 99')
    for c in g99.shapes:
        if c.has_text_frame:
            nfont += theme_font_all(c)
    notes.append('benchtop block: dx=%.2f dy=%.2f; non-theme font refs removed: %d' % (dx, dy, nfont))
    bench_bottom = max(find(list(slide.shapes), 'Picture 111').top / IN + find(list(slide.shapes), 'Picture 111').height / IN,
                       find(list(slide.shapes), 'Picture 110').top / IN + find(list(slide.shapes), 'Picture 110').height / IN)

    # ---------------------------------------------------------------- 5. Row 2
    Y2 = row1_bottom + 0.55
    LEFT_W = 26.0 - 0.5
    r2p = [P(HEAD_NEW, 40, True), P(SUB_NEW, 26)]
    r2h = measure(r2p, LEFT_W) / 0.85
    new_box(slide, 0.5, Y2, LEFT_W, r2h, 'Rev2 heading new measurements', r2p)
    fa_w = LEFT_W
    fa_h = fa_w * 7.5 / 20.0
    pa = slide.shapes.add_picture(FIG_A, Inches(0.5), Inches(Y2 + r2h + 0.1), Inches(fa_w), Inches(fa_h))
    pa.name = 'Rev2 Fig A sigma vs P'
    row2_bottom = Y2 + r2h + 0.1 + fa_h

    # ---------------------------------------------------------------- 6. Row 3
    Y3 = row2_bottom + 0.55
    cw = 7.0
    ch = cw * 6.0 / 7.0
    pc = slide.shapes.add_picture(FIG_C, Inches(0.5), Inches(Y3), Inches(cw), Inches(ch))
    pc.name = 'Rev2 Fig C pct increase vs T'
    cap = [P(CAP_C, 22)]
    new_box(slide, 0.5, Y3 + ch + 0.05, 7.6, 0.55, 'Rev2 Fig C caption', cap)
    row3_h = ch + 0.6
    fx, fw_ = 8.5, 11.2
    fpar = [P(FIND_HEAD, 28, True, after=6)] + [
        {'runs': [(t_, {'sz': 24, 'b': t_.startswith('→')})], 'after': 8, **({} if t_.startswith('→') else {'bullet': '•', 'marL': 0.35})}
        for t_ in FINDINGS]
    new_box(slide, fx, Y3, fw_, row3_h, 'Rev2 What we find box', fpar, fill='F3F3F3', line='B0B0B0', inset=(0.2, 0.15, 0.2, 0.15))
    mx = fx + fw_ + 0.4
    mw = 26.0 - mx
    mpar = [P(METH_HEAD, 28, True, after=6), P(METHOD, 22)]
    new_box(slide, mx, Y3, mw, row3_h, 'Rev2 Method box', mpar, fill='F3F3F3', line='B0B0B0', inset=(0.2, 0.15, 0.2, 0.15))
    row3_bottom = Y3 + row3_h

    # ---------------------------------------------------------------- 7. "Earlier work" block (right column, under benchtop)
    S2 = list(slide.shapes)
    s = 1.25
    SHIFT_R = 0.75           # draft inches: Adams/Larionov scans + labels moved right, clear of the compilation plot
    OLD_X0, OLD_Y0, OLD_W = 25.9, 33.18, 38.67 + SHIFT_R - 25.9
    YE = bench_bottom + 0.7
    ew = OLD_W * s
    X0 = RC_X + (RC_W - ew) / 2
    eh_p = [P(HEAD_EARLY, 32, True)]
    eh_h = measure(eh_p, RC_W) / 0.85
    hd = find(S2, 'TextBox 134')
    set_xywh(hd, RC_X, YE, RC_W, eh_h)
    set_body(hd, eh_p)
    register_fit('Earlier work heading', eh_p, RC_W, eh_h)
    sub = find(S2, 'TextBox 135')
    sp_ = [P('Recent experiments and modeling show decreasing σ with P', 24),
           P('(Pan et al. 2021, Psarakis et al. 2024)', 24)]
    sh_ = measure(sp_, RC_W) / 0.85
    set_xywh(sub, RC_X, YE + eh_h, RC_W, sh_)
    set_body(sub, sp_)
    register_fit('Earlier work subtitle', sp_, RC_W, sh_)
    Y0 = YE + eh_h + sh_ + 0.15

    def place(sh, ddx=0.0, ddy=0.0):
        x, y, w, h = geom(sh)
        set_xywh(sh, X0 + (x + ddx - OLD_X0) * s, Y0 + (y + ddy - OLD_Y0) * s, w * s, h * s)

    def rescale_fonts(sh, factor):
        for rpr in sh._element.iter(qn('a:rPr'), qn('a:endParaRPr')):
            sz = rpr.get('sz')
            rpr.set('sz', str(int(round((int(sz) if sz else 1800) * factor / 100.0) * 100)))
        theme_font_all(sh)

    for n, tx, ddx, ddy in [('Group 1', None, 0, 0), ('Picture 140', None, SHIFT_R, 0),
                            ('TextBox 141', None, SHIFT_R, 0.15), ('Group 136', None, SHIFT_R, 0),
                            ('TextBox 143', None, 0, 0), ('TextBox 176', None, 0, 0),
                            ('TextBox 177', None, SHIFT_R, -0.07)]:
        sh = find(S2, n, tx)
        place(sh, ddx, ddy)
        if sh.shape_type == 6:
            for c in sh.shapes:
                if c.has_text_frame:
                    rescale_fonts(c, s)
        elif sh.has_text_frame:
            rescale_fonts(sh, s)
    early_bottom = Y0 + (41.35 + 0.15 - OLD_Y0) * s

    # ---------------------------------------------------------------- 8. references footer
    ref_y = max(row3_bottom, early_bottom) + 0.45
    ref_h = 43.55 - ref_y
    refs_box = find(S2, 'TextBox 95')
    best = None
    for sz in range(18, 10, -1):
        runs = [('References: ', {'sz': sz, 'b': True})]
        for ent in REFS:
            for txt, b, i in ent:
                runs.append((txt, {'sz': sz, 'b': bool(b), 'i': bool(i)}))
        paras = [{'runs': runs}]
        if measure(paras, 43.0) <= MAX_USE * ref_h:
            best = (sz, paras)
            break
    assert best, 'references do not fit'
    ref_h = min(ref_h, measure(best[1], 43.0) / 0.80)
    ref_y = 43.55 - ref_h
    set_xywh(refs_box, 0.5, ref_y, 43.0, ref_h)
    set_body(refs_box, best[1])
    register_fit('References', best[1], 43.0, ref_h)
    notes.append('references: %d pt (draft: default 18 pt), box y %.2f-%.2f' % (best[0], ref_y, ref_y + ref_h))

    # ---------------------------------------------------------------- 9. separators (clean, aligned to gutters)
    sep_y1 = Y1 - 0.25
    add_line(slide, 0.5, sep_y1, 43.5, sep_y1, 'Rev2 separator header')
    add_line(slide, 13.75, Y1, 13.75, row1_bottom, 'Rev2 separator col1|col2')
    add_line(slide, 0.5, row1_bottom + 0.27, 26.0, row1_bottom + 0.27, 'Rev2 separator row1|row2')
    add_line(slide, 0.5, row2_bottom + 0.27, 26.0, row2_bottom + 0.27, 'Rev2 separator row2|row3')
    add_line(slide, 26.24, Y1, 26.24, max(row3_bottom, early_bottom), 'Rev2 separator left|right')
    add_line(slide, 0.5, ref_y - 0.2, 43.5, ref_y - 0.2, 'Rev2 separator footer')

    prs.save(DST)
    geo = dict(Y1=Y1, row1_bottom=row1_bottom, Y2=Y2, row2_bottom=row2_bottom, Y3=Y3, row3_bottom=row3_bottom,
               bench_bottom=bench_bottom, YE=YE, early_bottom=early_bottom, ref_y=ref_y, header_bottom=header_bottom,
               title_pt=tsz, title_lines=tlines, left_bottom=left_bottom, hero_bottom=hero_bottom)
    return removed, notes, geo


# --------------------------------------------------------------------------- verification + preview
def walk(shapes, tf=(0.0, 0.0, 1.0, 1.0), path=''):
    """Yield (path, shape, abs_x, abs_y, abs_w, abs_h) in inches for every shape (groups resolved)."""
    for sh in shapes:
        ox, oy, sx, sy = tf
        x = ox + sh.left / IN * sx
        y = oy + sh.top / IN * sy
        w = sh.width / IN * sx
        h = sh.height / IN * sy
        yield path, sh, x, y, w, h
        if sh.shape_type == 6:
            xf = sh._element.grpSpPr.find(qn('a:xfrm'))
            ch, ce = xf.find(qn('a:chOff')), xf.find(qn('a:chExt'))
            csx = w / (int(ce.get('cx')) / IN)
            csy = h / (int(ce.get('cy')) / IN)
            ntf = (x - int(ch.get('x')) / IN * csx, y - int(ch.get('y')) / IN * csy, csx, csy)
            yield from walk(sh.shapes, ntf, path + sh.name + '/')


def verify(md5_before):
    prs = Presentation(DST)
    out = []
    out.append('slides: %d  size: %.1f x %.1f in' % (len(prs.slides), prs.slide_width / IN, prs.slide_height / IN))
    md5_after = hashlib.md5(open(SRC, 'rb').read()).hexdigest()
    out.append('draft md5 before %s after %s -> %s' % (md5_before, md5_after, 'UNCHANGED' if md5_before == md5_after else 'CHANGED!'))
    items = [(p, s, x, y, w, h) for p, s, x, y, w, h in walk(prs.slides[0].shapes)]
    bad = [(p + s.name, x, y, w, h) for p, s, x, y, w, h in items if x < -1e-6 or y < -1e-6 or x + w > 44 + 1e-6 or y + h > 44 + 1e-6]
    out.append('shapes outside 0-44 in: %s' % (bad or 'none'))
    mm = [(p + s.name, round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)) for p, s, x, y, w, h in items
          if x < 0.4 - 1e-6 or y < 0.4 - 1e-6 or x + w > 43.6 + 1e-6 or y + h > 43.6 + 1e-6]
    out.append('shapes inside the 0.4 in outer margin violation: %s' % (mm or 'none'))
    leaves = [i for i in items if i[1].shape_type != 6 and i[4] * i[5] > 0]
    ov = []
    for a in range(len(leaves)):
        for b in range(a + 1, len(leaves)):
            pa, sa, xa, ya, wa, ha = leaves[a]
            pb, sb, xb, yb, wb, hb = leaves[b]
            if pa and pa == pb:
                pass
            ix = min(xa + wa, xb + wb) - max(xa, xb)
            iy = min(ya + ha, yb + hb) - max(ya, yb)
            if ix > 0 and iy > 0 and ix * iy > 0.05:
                ov.append((round(ix * iy, 2), pa + sa.name, pb + sb.name))
    ov.sort(reverse=True)
    out.append('overlaps > 0.05 in^2 (%d):' % len(ov))
    for o in ov:
        out.append('   %.2f  %s  <>  %s' % o)
    return out, items


def fit_report():
    lines = []
    fails = 0
    for name, est, h, mn in FITS:
        use = est / h
        flag = 'FAIL' if use > MAX_USE else 'ok'
        fails += flag == 'FAIL'
        lines.append('  %-34s est %.2f in / box %.2f in = %3.0f%%  min %2.0f pt  %s' % (name, est, h, 100 * use, mn, flag))
    return lines, fails


def preview(items, path_png, width=6600):
    """Schematic preview: real images at their positions, text boxes drawn with DejaVu Sans (SUBSTITUTE for Aptos)."""
    S = width / 44.0
    img = Image.new('RGB', (width, width), 'white')
    dr = ImageDraw.Draw(img)
    names = {}
    pj = os.path.join(RECON, 'poster_shapes.json')
    if os.path.exists(pj):
        def lv(l):
            for s in l:
                yield s
                yield from lv(s.get('children', []))
        for s in lv(json.load(open(pj))['slides'][0]['shapes']):
            if s.get('preview_png'):
                names[s['name']] = os.path.join(RECON, 'images', s['preview_png'].replace('_embedded_bitmap_preview', '_preview'))
    for p, sh, x, y, w, h in items:
        X, Y, W, H = x * S, y * S, w * S, h * S
        if sh.shape_type == 6:
            continue
        if sh.shape_type == 13:                      # picture
            try:
                if sh.image.content_type in ('image/x-emf', 'image/emf'):
                    raise ValueError
                import io
                im = Image.open(io.BytesIO(sh.image.blob)).convert('RGBA')
                cl, cr, ct, cb = sh.crop_left, sh.crop_right, sh.crop_top, sh.crop_bottom
                iw, ih = im.size
                im = im.crop((int(cl * iw), int(ct * ih), int(iw - cr * iw), int(ih - cb * ih)))
            except Exception:
                fp = names.get(sh.name)
                im = Image.open(fp).convert('RGBA') if fp and os.path.exists(fp) else Image.new('RGBA', (50, 50), (220, 220, 220, 255))
            im = im.resize((max(1, int(W)), max(1, int(H))), Image.LANCZOS)
            img.paste(im, (int(X), int(Y)), im)
            dr.rectangle([X, Y, X + W, Y + H], outline=(200, 0, 0), width=2)
        elif sh.shape_type is not None and 'LINE' in str(sh.shape_type) or sh.name.startswith('Rev2 separator'):
            dr.line([X, Y, X + W, Y + H], fill=(110, 110, 110), width=5)
        elif sh.has_text_frame:
            dr.rectangle([X, Y, X + W, Y + H], outline=(0, 0, 170), width=2)
            ty = Y + 0.05 * S
            for para in sh.text_frame.paragraphs:
                runs = [r for r in para.runs if r.text]
                if not runs:
                    ty += 18 * 1.2 / 72 * S
                    continue
                sz = max((r.font.size.pt if r.font.size else 18) for r in runs)
                bold = sum(len(r.text) for r in runs if r.font.bold) * 2 > sum(len(r.text) for r in runs)
                txt = ''.join(r.text for r in runs)
                ppr = para._p.find(qn('a:pPr'))
                ml = int(ppr.get('marL', 0)) / IN if ppr is not None else 0
                f = pil_font(sz, bold, scale=S / 72.0)
                lines = wrap_lines(txt, sz, bold, (w - 0.2 - ml) * 72.0) if sh.text_frame.word_wrap is not False else [txt]
                for ln in lines:
                    dr.text((X + (0.1 + ml) * S, ty), ln, font=f, fill=(0, 0, 0))
                    ty += sz * 1.2 / 72 * S
                sa = para._p.find(qn('a:pPr') + '/' + qn('a:spcAft'))
                if sa is not None:
                    ty += int(sa[0].get('val')) / 100.0 / 72 * S
        elif sh.shape_type == 1:                      # autoshape (none kept) / fallback
            dr.rectangle([X, Y, X + W, Y + H], outline=(120, 120, 120), width=2)
    img.save(path_png.replace('.png', '_full6600.png') if False else path_png)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', default=TITLE, help='official AGU title (default: placeholder)')
    ap.add_argument('--no-preview', action='store_true')
    args = ap.parse_args()
    md5_before = hashlib.md5(open(SRC, 'rb').read()).hexdigest()
    removed, notes, geo = build(args.title)
    print('saved', DST)
    for n in notes:
        print(' ', n)
    print('geometry:', json.dumps({k: round(v, 2) for k, v in geo.items()}))
    out, items = verify(md5_before)
    print('\n'.join(out))
    fl, fails = fit_report()
    print('text fit (DejaVu Sans substitute; >%.0f%% = FAIL):' % (100 * MAX_USE))
    print('\n'.join(fl))
    print('removed shapes: %d' % len(removed))
    if not args.no_preview:
        img = preview(items, os.path.join(HERE, 'poster_rev2_full_hi.png'))
        W = 6600
        small = img.resize((3300, 3300), Image.LANCZOS)
        small.save(os.path.join(HERE, 'poster_rev2_preview.png'))
        for nm, (a, b) in {'top': (0, 1700), 'middle': (1700, 3400), 'bottom': (3400, 5100)}.items():
            pass
        # three horizontal crops at full 6600 px width (150 px/in)
        bands = {'top': (0, 15.5), 'middle': (14.5, 30.0), 'bottom': (29.0, 44.0)}
        for nm, (y0, y1) in bands.items():
            img.crop((0, int(y0 * 150), W, int(y1 * 150))).save(os.path.join(HERE, 'poster_rev2_%s.png' % nm))
        os.remove(os.path.join(HERE, 'poster_rev2_full_hi.png'))
    if fails:
        print('TEXT FIT FAILURES: %d' % fails)
        sys.exit(2)


if __name__ == '__main__':
    main()
