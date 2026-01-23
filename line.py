import streamlit as st
import pandas as pd
import os
import io
import re
import math
import datetime
from io import BytesIO

# --- PDF & Image Imports ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# --- QR Code Import ---
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- STYLE DEFINITIONS (EXACTLY AS PER YOUR DESIGN) ---
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)

bin_bold_style = ParagraphStyle('BinBold', fontName='Helvetica-Bold', fontSize=24, alignment=TA_CENTER)
bin_desc_style = ParagraphStyle('BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER)
bin_qty_style = ParagraphStyle('BinQty', fontName='Helvetica-Bold', fontSize=28, alignment=TA_CENTER)

rl_header_style = ParagraphStyle('RLHeader', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle('RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)
location_header_style = ParagraphStyle('LocHeader', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER)

master_val_style_left = ParagraphStyle('MasterValLeft', fontName='Helvetica-Bold', fontSize=13, alignment=TA_LEFT)
master_val_style_center = ParagraphStyle('MasterValCenter', fontName='Helvetica-Bold', fontSize=13, alignment=TA_CENTER)

# --- FORMATTING HELPERS (RACK LABELS) ---
def format_part_no_v1(part_no):
    return Paragraph(f"<b><font size=22>{part_no}</font></b>", ParagraphStyle('P1', fontName='Helvetica-Bold', alignment=TA_LEFT))

def format_part_no_v2(part_no):
    if not part_no or str(part_no).upper() == 'EMPTY':
        return Paragraph("<b><font size=34>EMPTY</font></b>", bold_style_v2)
    part_no = str(part_no)
    if len(part_no) > 5:
        p1, p2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{p1}</font><font size=40>{p2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description_v1(desc):
    return Paragraph(str(desc), ParagraphStyle('D1', fontName='Helvetica', fontSize=12, alignment=TA_LEFT))

def format_description(desc):
    return Paragraph(str(desc), desc_style)

def get_dynamic_location_style(text, col_type='Default'):
    size = 18 if col_type == 'Bus Model' else 20
    return ParagraphStyle('DynLoc', fontName='Helvetica-Bold', fontSize=size, alignment=TA_CENTER)

# --- DATA EXTRACTION HELPERS ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    def find_key(keywords):
        for k in cols:
            if all(word in k for word in keywords): return cols[k]
        return None
    
    return {
        'Part No': find_key(['PART', 'NO']) or find_key(['PART', 'NUM']),
        'Description': find_key(['DESC']),
        'Bus Model': find_key(['BUS', 'MODEL']),
        'Station No': find_key(['STATION', 'NO']) or find_key(['STATION', '#']),
        'Station Name': find_key(['STATION', 'NAME']),
        'Container': find_key(['CONTAINER']),
        'Qty/Bin': find_key(['QTY', 'BIN']),
        'Qty/Veh': find_key(['QTY', 'VEH']),
        'Zone': find_key(['ZONE'])
    }

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def extract_store_location_data_from_excel(row_data):
    col_lookup = {str(k).strip().upper(): k for k in row_data.keys()}
    def get_val(names):
        for n in names:
            if n.upper() in col_lookup: return str(row_data.get(col_lookup[n.upper()])).strip()
        return ""
    return [get_val(['ST. NAME (Short)']), get_val(['Store Location']), get_val(['ABB ZONE']), get_val(['ABB LOCATION']), get_val(['ABB FLOOR']), get_val(['ABB RACK NO']), get_val(['ABB LEVEL IN RACK'])]

# --- CORE ASSIGNMENT LOGIC (STATION-WISE RESET) ---
def automate_location_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    col_map = find_required_columns(df)
    df_processed = df.copy()
    rename_dict = {v: k for k, v in col_map.items() if v}
    df_processed.rename(columns=rename_dict, inplace=True)
    
    df_processed['bin_area'] = df_processed['Container'].apply(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
    df_processed['bins_per_cell'] = df_processed['Container'].apply(lambda x: bin_info_map.get(x, {}).get('capacity', 1))
    
    final_data = []
    for st_no, st_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {st_no}...")
        st_name = st_group['Station Name'].iloc[0] if 'Station Name' in st_group.columns else st_no
        
        container_groups = sorted(st_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        cells_needed = sum([math.ceil(len(g) / g['bins_per_cell'].iloc[0]) for _, g in container_groups])
        
        racks_needed = math.ceil(cells_needed / (len(levels) * cells_per_level))
        avail_locs = []
        for r_idx in range(1, racks_needed + 1):
            r_str = f"{r_idx:02d}"
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    avail_locs.append({'Rack No 1st': r_str[0], 'Rack No 2nd': r_str[1], 'Level': lvl, 'P_Cell': f"{c_idx:02d}", 'Rack': base_rack_id})

        ptr = 0
        for _, c_df in container_groups:
            parts = c_df.to_dict('records')
            cap = parts[0]['bins_per_cell']
            for i in range(0, len(parts), cap):
                chunk = parts[i:i+cap]
                if ptr < len(avail_locs):
                    loc = avail_locs[ptr]
                    for p in chunk:
                        p.update(loc); final_data.append(p)
                    ptr += 1
        
        for i in range(ptr, len(avail_locs)):
            empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': st_group['Bus Model'].iloc[0], 'Station No': st_no, 'Station Name': st_name}
            empty.update(avail_locs[i]); final_data.append(empty)
            
    res = pd.DataFrame(final_data)
    # Assign sequential Cell IDs (1, 2, 3...) within each Rack/Level
    res['Cell'] = res.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level']).cumcount() + 1
    res['Cell'] = res['Cell'].apply(lambda x: f"{x:02d}")
    return res

# --- PDF GENERATION: RACK LABELS V1 (MULTIPLE PARTS) ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df_parts = df[df['Part No'] != 'EMPTY'].copy()
    df_parts['loc_key'] = df_parts.apply(lambda r: f"{r['Station No']}-{r['Rack No 1st']}{r['Rack No 2nd']}-{r['Level']}-{r['P_Cell']}", axis=1)
    grouped = df_parts.groupby('loc_key')
    
    for i, (key, group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int((i / len(grouped)) * 100))
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        p1 = group.iloc[0].to_dict()
        p2 = group.iloc[1].to_dict() if len(group) > 1 else p1
        
        t1 = Table([['Part No', format_part_no_v1(p1['Part No'])], ['Description', format_description_v1(p1['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        t2 = Table([['Part No', format_part_no_v1(p2['Part No'])], ['Description', format_description_v1(p2['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        loc_vals = extract_location_values(p1)
        f_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v)) for v in loc_vals]
        lt = Table([f_loc], colWidths=[4*cm, 1.8*cm, 2.7*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm], rowHeights=0.8*cm)
        
        style = TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (0,-1), 'CENTER')])
        t1.setStyle(style); t2.setStyle(style)
        
        l_colors = [colors.white, colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        l_style = [('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        for j, c in enumerate(l_colors): l_style.append(('BACKGROUND', (j, 0), (j, 0), c))
        lt.setStyle(TableStyle(l_style))
        
        elements.extend([t1, Spacer(1, 0.3*cm), t2, Spacer(1, 0.3*cm), lt, Spacer(1, 1.2*cm)])
    
    doc.build(elements); buffer.seek(0)
    return buffer, {}

# --- PDF GENERATION: RACK LABELS V2 (SINGLE PART) ---
def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df_parts = df[df['Part No'] != 'EMPTY'].copy()
    
    for i, row in enumerate(df_parts.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / len(df_parts)) * 100))
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        t = Table([['Part No', format_part_no_v2(row['Part No'])], ['Description', format_description(row['Description'])]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_vals = extract_location_values(row)
        f_loc = [Paragraph('Line Location', location_header_style)] + [Paragraph(v, get_dynamic_location_style(v)) for v in loc_vals]
        lt = Table([f_loc], colWidths=[4*cm, 1.7*cm, 2.9*cm, 1.3*cm, 1.2*cm, 1.3*cm, 1.3*cm, 1.3*cm], rowHeights=0.9*cm)
        
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('ALIGN', (0,0), (0,-1), 'CENTER')]))
        l_colors = [colors.white, colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        l_style = [('GRID', (0,0), (-1,-1), 1, colors.black), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]
        for j, c in enumerate(l_colors): l_style.append(('BACKGROUND', (j, 0), (j, 0), c))
        lt.setStyle(TableStyle(l_style))
        
        elements.extend([t, Spacer(1, 0.3*cm), lt, Spacer(1, 1.5*cm)])
        
    doc.build(elements); buffer.seek(0)
    return buffer, {}

# --- PDF GENERATION: BIN LABELS ---
def generate_qr_code_image(data):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(box_size=10, border=4); qr.add_data(data); qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return RLImage(buf, width=2.5*cm, height=2.5*cm)

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    STICKER_W, STICKER_H = 10*cm, 15*cm
    CONTENT_W, CONTENT_H = 10*cm, 7.2*cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_W, STICKER_H), topMargin=0.2*cm, bottomMargin=STICKER_H-CONTENT_H-0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    df_f = df[df['Part No'] != 'EMPTY'].copy()
    elements = []

    def draw_border(canvas, doc):
        canvas.saveState(); canvas.setLineWidth(1.8)
        canvas.rect((STICKER_W-CONTENT_W)/2+0.1*cm, STICKER_H-CONTENT_H-0.2*cm, CONTENT_W-0.2*cm, CONTENT_H)
        canvas.restoreState()

    for i, row in enumerate(df_f.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i/len(df_f))*100))
        cw = CONTENT_W - 0.2*cm
        
        main_t = Table([["Part No", Paragraph(str(row['Part No']), bin_bold_style)], ["Description", Paragraph(str(row['Description'])[:47], bin_desc_style)], ["Qty/Bin", Paragraph(str(row.get('Qty/Bin', '1')), bin_qty_style)]], colWidths=[cw/3, cw*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        
        iw = (cw * 2/3)/7
        s_inner = Table([extract_store_location_data_from_excel(row)], colWidths=[iw]*7, rowHeights=[0.5*cm])
        s_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        s_row = Table([[Paragraph("Store Location", bin_desc_style), s_inner]], colWidths=[cw/3, cw*2/3], rowHeights=[0.5*cm])
        s_row.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black)]))
        
        l_inner = Table([extract_location_values(row)], colWidths=[iw]*7, rowHeights=[0.5*cm])
        l_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        l_row = Table([[Paragraph("Line Location", bin_desc_style), l_inner]], colWidths=[cw/3, cw*2/3], rowHeights=[0.5*cm])
        l_row.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black)]))

        mtm_t = None
        if mtm_models:
            q_v = [Paragraph(f"<b>{row.get('Qty/Veh','1')}</b>", bin_qty_style) if str(row['Bus Model']).upper()==m.upper() else "" for m in mtm_models]
            mtm_t = Table([mtm_models, q_v], colWidths=[(3.6*cm)/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm, 0.75*cm])
            mtm_t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        qr = generate_qr_code_image(f"{row['Part No']}|{row['Station No']}|{row['Rack']}{row['Rack No 1st']}")
        bot = Table([[mtm_t or "", "", qr or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, cw-7.1*cm], rowHeights=[2.5*cm])
        
        elements.extend([main_t, s_row, l_row, Spacer(1, 0.2*cm), bot, PageBreak()])

    doc.build(elements[:-1], onFirstPage=draw_border, onLaterPages=draw_border); buffer.seek(0)
    return buffer, {}

# --- PDF GENERATION: RACK LIST ---
def generate_rack_list_pdf(df, base_rack_id, top_logo, logo_w, logo_h, fixed_logo_path, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df_parts = df[df['Part No'] != 'EMPTY'].copy()
    grouped = df_parts.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd'])

    for i, ((st_no, r1, r2), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int((i/len(grouped))*100))
        r_key = f"{r1}{r2}"
        
        lg = ""
        if top_logo: lg = RLImage(io.BytesIO(top_logo.getvalue()), width=logo_w*cm, height=logo_h*cm)
        elements.append(Table([[Paragraph("Document Ref No.:", rl_header_style), "", lg]], colWidths=[5*cm, 17.5*cm, 5*cm]))
        
        m_data = [
            [Paragraph("STATION NAME", master_val_style_left), Paragraph(str(group.iloc[0]['Station Name']), master_val_style_left), Paragraph("STATION NO", master_val_style_left), Paragraph(str(st_no), master_val_style_center)],
            [Paragraph("MODEL", master_val_style_left), Paragraph(str(group.iloc[0]['Bus Model']), master_val_style_left), Paragraph("RACK NO", master_val_style_left), Paragraph(f"Rack - {r_key}", master_val_style_center)]
        ]
        mt = Table(m_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        mt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        elements.append(mt)

        h = ["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]
        w = [1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        if 'Zone' in group.columns and group['Zone'].any():
            h.insert(0, "ZONE"); w = [2.0*cm, 1.3*cm, 4.0*cm, 8.2*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        
        rows = [h]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{st_no}-{base_rack_id}{r_key}-{r['Level']}{r['Cell']}"
            row = [str(idx+1), r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r.get('Qty/Bin', '1'), loc]
            if 'Zone' in h: row.insert(0, r.get('Zone', ''))
            rows.append(row)

        dt = Table(rows, colWidths=w)
        dt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        elements.extend([dt, PageBreak()])

    doc.build(elements); buffer.seek(0)
    return buffer, len(grouped)

# --- MAIN UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    
    st.sidebar.title("📄 Options")
    out_type = st.sidebar.selectbox("Output Type", ["Rack Labels", "Bin Labels", "Rack List"])
    rack_fmt = "Single Part"
    if out_type == "Rack Labels": rack_fmt = st.sidebar.selectbox("Format", ["Single Part", "Multiple Parts"])
    
    base_id = st.sidebar.text_input("Infrastructure ID (e.g. R, TR)", "R")
    
    # Logo and MTM Logic
    mtm = []
    top_logo, logo_w, logo_h = None, 4.0, 1.5
    if out_type == "Bin Labels":
        m1 = st.sidebar.text_input("Model 1", "7M"); m2 = st.sidebar.text_input("Model 2", "9M")
        mtm = [m.strip() for m in [m1, m2] if m.strip()]
    elif out_type == "Rack List":
        top_logo = st.sidebar.file_uploader("Upload Logo", type=['png', 'jpg'])
        logo_w = st.sidebar.slider("Logo Width", 1.0, 8.0, 4.0)

    uploaded = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv'])

    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        col_check = find_required_columns(df)
        
        if col_check['Station No'] and col_check['Container']:
            lvls = st.sidebar.multiselect("Levels", ['A','B','C','D','E','F'], ['A','B','C','D'])
            num_c = st.sidebar.number_input("Cells per Level", 1, 50, 10)
            
            unique_c = sorted(df[col_check['Container']].dropna().unique())
            bin_map = {}
            for c in unique_c:
                c1, c2 = st.sidebar.columns(2)
                d = c1.text_input(f"{c} Dim", "600x400", key=f"d_{c}")
                cp = c2.number_input(f"Cap", 1, 10, key=f"c_{c}")
                nums = [int(n) for n in re.findall(r'\d+', d)]
                bin_map[c] = {'dims': (nums[0], nums[1]) if len(nums)>=2 else (0,0), 'capacity': cp}

            if st.button("🚀 Generate PDF"):
                status = st.empty()
                df_loc = automate_location_assignment(df, base_id, lvls, num_c, bin_map, status)
                
                if out_type == "Rack Labels":
                    func = generate_rack_labels_v2 if rack_fmt=="Single Part" else generate_rack_labels_v1
                    buf, _ = func(df_loc, st.progress(0))
                elif out_type == "Bin Labels":
                    buf, _ = generate_bin_labels(df_loc, mtm, st.progress(0))
                else:
                    buf, _ = generate_rack_list_pdf(df_loc, base_id, top_logo, logo_w, logo_h, "", st.progress(0))
                
                st.download_button("📥 Download PDF", buf.getvalue(), f"{out_type}.pdf")
                status.empty()
        else:
            st.error("❌ Required columns (Station and Container) not found.")

if __name__ == "__main__":
    main()
