import streamlit as st
import pandas as pd
import os
import io
import re
import math
import datetime
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# QR Code imports
try:
    import qrcode
    from PIL import Image
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="AgiloSmartTag Studio",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions ---
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32, spaceBefore=0, spaceAfter=2, wordWrap='CJK'
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)
bin_bold_style = ParagraphStyle(
    name='BinBold', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER, leading=20
)
bin_desc_style = ParagraphStyle(
    name='BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER, leading=12
)
bin_qty_style = ParagraphStyle(
    name='BinQty', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER, leading=16
)
rl_header_style = ParagraphStyle(
    name='RLHeader', fontName='Helvetica-Bold', fontSize=11, alignment=TA_LEFT
)
rl_cell_left_style = ParagraphStyle(
    name='RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT
)

# --- Formatting Functions ---
def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- Core Logic Functions ---
def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    part_no_key = next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)
    desc_key = next((k for k in cols if 'DESC' in k), None)
    bus_model_key = next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)
    station_no_key = next((k for k in cols if 'STATION' in k and 'NO' in k), None)
    if not station_no_key:
        station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    return {
        'Part No': cols.get(part_no_key),
        'Description': cols.get(desc_key),
        'Bus Model': cols.get(bus_model_key),
        'Station No': cols.get(station_no_key),
        'Container': cols.get(container_type_key)
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def generate_by_rack_type(df, base_rack_id, rack_templates, status_text=None):
    req = find_required_columns(df)
    df_p = df.copy()
    df_p.rename(columns={req['Part No']: 'Part No', req['Description']: 'Description', 
                         req['Bus Model']: 'Bus Model', req['Station No']: 'Station No', 
                         req['Container']: 'Container'}, inplace=True)
    
    final_data = []
    if not rack_templates: return pd.DataFrame()
    
    # Using first template for the logic
    template_name = list(rack_templates.keys())[0]
    config = rack_templates[template_name]
    levels = config['levels']

    for station_no, station_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Allocating Station: {station_no}...")
        curr_rack_num = 1
        curr_lvl_idx = 0
        curr_cell_idx = 1
        
        for cont_type, parts_group in station_group.groupby('Container', sort=True):
            bins_per_level = config['capacities'].get(cont_type, 1)
            all_parts = parts_group.to_dict('records')

            for part in all_parts:
                if curr_cell_idx > bins_per_level:
                    curr_cell_idx = 1
                    curr_lvl_idx += 1
                
                if curr_lvl_idx >= len(levels):
                    curr_lvl_idx = 0
                    curr_rack_num += 1
                    curr_cell_idx = 1

                rack_str = f"{curr_rack_num:02d}"
                part.update({
                    'Rack': base_rack_id, 'Rack No 1st': rack_str[0], 'Rack No 2nd': rack_str[1],
                    'Level': levels[curr_lvl_idx], 'Physical_Cell': f"{curr_cell_idx:02d}",
                    'Station No': station_no, 'Rack Key': rack_str
                })
                final_data.append(part)
                curr_cell_idx += 1
    return pd.DataFrame(final_data)

def assign_sequential_location_ids(df):
    if df.empty: return df
    df_sorted = df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Physical_Cell']).copy()
    df_parts_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    
    location_counters = {}
    sequential_ids = []
    
    for _, row in df_parts_only.iterrows():
        key = (row['Station No'], row['Rack No 1st'], row['Rack No 2nd'], row['Level'])
        if key not in location_counters: location_counters[key] = 1
        sequential_ids.append(location_counters[key])
        location_counters[key] += 1
        
    df_parts_only['Cell'] = sequential_ids
    df_empty_only = df_sorted[df_sorted['Part No'].astype(str).str.upper() == 'EMPTY'].copy()
    df_empty_only['Cell'] = df_empty_only['Physical_Cell']
    
    return pd.concat([df_parts_only, df_empty_only], ignore_index=True)

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def extract_store_location_data_from_excel(row):
    def get_clean_value(possible_names, default=''):
        for name in possible_names:
            val = row.get(name)
            if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                return str(val).strip()
        return default

    store_location = get_clean_value(['Store Location', 'STORELOCATION', 'Store_Location'])
    zone = get_clean_value(['ABB ZONE', 'ABB_ZONE', 'ABBZONE', 'Zone'])
    location = get_clean_value(['ABB LOCATION', 'ABB_LOCATION', 'ABBLOCATION'])
    floor = get_clean_value(['ABB FLOOR', 'ABB_FLOOR', 'ABBFLOOR'])
    rack_no = get_clean_value(['ABB RACK NO', 'ABB_RACK_NO', 'ABBRACKNO'])
    level_in_rack = get_clean_value(['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK', 'ABBLEVELINRACK'])
    station_name_short = get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'Station Name Short']) 
    
    return [store_location, station_name_short, zone, location, floor, rack_no, level_in_rack]

def generate_qr_code_image(data):
    if not QR_AVAILABLE: return None
    try:
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)
    except: return None

# --- PDF Generators ---
def generate_rack_labels(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_parts_only = df[df['Part No'].astype(str).str.upper() != 'EMPTY'].copy()
    total_labels = len(df_parts_only)
    label_count = 0

    for i, part in enumerate(df_parts_only.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part_table = Table([['Part No', format_part_no_v2(str(part['Part No']))], ['Description', format_description(str(part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        location_values = extract_location_values(part)
        location_data = [['Line Location'] + location_values]
        col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths) for w in col_widths]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=1.2*cm)
        part_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0, 0), (0, -1), 16)]))
        location_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        location_style = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTSIZE', (0, 0), (-1, -1), 16)]
        for j, color in enumerate(location_colors): location_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(location_style))
        elements.append(part_table); elements.append(Spacer(1, 0.3 * cm)); elements.append(location_table); elements.append(Spacer(1, 0.2 * cm))
        label_count += 1
    if elements: doc.build(elements)
    buffer.seek(0)
    summary = df_parts_only.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd']).size().reset_index(name='Labels')
    summary['Rack'] = summary['Rack No 1st'] + summary['Rack No 2nd']
    return buffer, summary[['Station No', 'Rack', 'Labels']]

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE:
        st.error("❌ QR Code library not found.")
        return None, {}
    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 8.5 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total_labels = len(df_filtered)
    label_summary, all_elements = {}, []
    def draw_border(canvas, doc):
        canvas.saveState()
        canvas.setLineWidth(1.8)
        canvas.rect((STICKER_WIDTH-CONTENT_BOX_WIDTH)/2 + doc.leftMargin, STICKER_HEIGHT-CONTENT_BOX_HEIGHT-0.2*cm, CONTENT_BOX_WIDTH-0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()
    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        part_no, desc, qty_bin, qty_veh = str(row.get('Part No', '')), str(row.get('Description', '')), str(row.get('Qty/Bin', '')), str(row.get('Qty/Veh', ''))
        store_loc_raw = extract_store_location_data_from_excel(row)
        line_loc_raw = extract_location_values(row)
        qr_data = f"Part No: {part_no}\nDesc: {desc}\nQty/Bin: {qty_bin}\nQty/Veh: {qty_veh}\nStore: {'|'.join(store_loc_raw)}\nLine: {'|'.join(line_loc_raw)}"
        qr_image = generate_qr_code_image(qr_data)
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        
        station_header = Table([[Paragraph("STATION NAME (SHORT)", bin_desc_style), Paragraph(store_loc_raw[1], bin_bold_style)]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.8*cm])
        station_header.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('BACKGROUND', (0,0), (0,0), colors.lightgrey)]))
        main_table = Table([["Part No", Paragraph(part_no, bin_bold_style)], ["Description", Paragraph(desc[:47], bin_desc_style)], ["Qty/Bin", Paragraph(qty_bin, bin_qty_style)]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        inner_w = content_width * 2/3
        col_props = [1.8, 2.4, 0.7, 0.7, 0.7, 0.7, 0.9]
        inner_cols = [w * inner_w / sum(col_props) for w in col_props]
        
        st_loc_row = Table([[Paragraph("Store Location", bin_desc_style), Table([store_loc_raw], colWidths=inner_cols, rowHeights=[0.5*cm])]], colWidths=[content_width/3, inner_w])
        st_loc_row.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        ln_loc_row = Table([[Paragraph("Line Location", bin_desc_style), Table([line_loc_raw], colWidths=inner_cols, rowHeights=[0.5*cm])]], colWidths=[content_width/3, inner_w])
        ln_loc_row.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        mtm_table = None
        if mtm_models:
            mtm_qty = [Paragraph(f"<b>{qty_veh}</b>", bin_qty_style) if str(row.get('Bus Model','')).strip().upper() == m.upper() else "" for m in mtm_models]
            mtm_table = Table([mtm_models, mtm_qty], colWidths=[(3.6*cm)/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
            
        bottom_row = Table([[mtm_table or "", "", qr_image or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, content_width-7.1*cm], rowHeights=[2.5*cm])
        all_elements.extend([station_header, main_table, st_loc_row, ln_loc_row, Spacer(1, 0.2*cm), bottom_row, PageBreak()])
    if all_elements: doc.build(all_elements[:-1], onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements, df = [], df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df['Rack Key'] = df['Rack No 1st'] + df['Rack No 2nd']
    grouped = df.groupby(['Station No', 'Rack Key'])
    for i, ((st_no, r_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1)/len(grouped))*100))
        first = group.iloc[0]
        st_name = str(first.get('ST. NAME (Short)', ''))
        
        master_table = Table([[Paragraph("STATION NAME", ParagraphStyle('H', fontName='Helvetica-Bold')), Paragraph(st_name, ParagraphStyle('V', fontName='Helvetica-Bold', fontSize=13)), Paragraph("STATION NO", ParagraphStyle('H', fontName='Helvetica-Bold')), Paragraph(str(st_no), ParagraphStyle('V', fontName='Helvetica-Bold', fontSize=13))],
                              [Paragraph("MODEL", ParagraphStyle('H', fontName='Helvetica-Bold')), Paragraph(str(first.get('Bus Model','')), ParagraphStyle('V', fontName='Helvetica-Bold', fontSize=13)), Paragraph("RACK NO", ParagraphStyle('H', fontName='Helvetica-Bold')), Paragraph(f"Rack - {r_key}", ParagraphStyle('V', fontName='Helvetica-Bold', fontSize=13))]], colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm]*2)
        master_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        
        data = [["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r.get('Bus Model')}-{r.get('Station No')}-{base_rack_id}{r_key}-{r.get('Level')}{r.get('Cell')}"
            data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r['Qty/Bin'], loc])
        
        t = Table(data, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        t.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        elements.extend([master_table, Spacer(1, 0.2*cm), t, PageBreak()])
    doc.build(elements[:-1])
    buffer.seek(0)
    return buffer, len(grouped)

# --- Main App ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed by Rimjhim Rani | Agilomatrix</p>", unsafe_allow_html=True)
    
    st.sidebar.title("📄 Config")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    
    # Model Setup for Bin Labels
    m1, m2, m3 = "7M", "9M", "12M"
    if output_type == "Bin Labels":
        m1 = st.sidebar.text_input("Model 1", "7M")
        m2 = st.sidebar.text_input("Model 2", "9M")
        m3 = st.sidebar.text_input("Model 3", "12M")

    base_rack_id = st.sidebar.text_input("Infrastructure ID", "R")
    uploaded_file = st.file_uploader("Upload Your Data Here (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {len(df)} parts.")
        req = find_required_columns(df)
        
        if req['Container'] and req['Station No']:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Rack Type Configuration")
            unique_c = get_unique_containers(df, req['Container'])
            
            # Simplified for single Rack Type Template
            r_levels = st.sidebar.multiselect("Levels", ['A','B','C','D','E','F'], default=['A','B','C','D'])
            st.sidebar.caption("Define Bins per Shelf:")
            caps = {c: st.sidebar.number_input(f"{c} per shelf", 1, 50, 4) for c in unique_c}
            rack_templates = {"Default": {'levels': r_levels, 'capacities': caps}}

            if st.button("🚀 Generate PDF", type="primary"):
                status = st.empty()
                df_a = generate_by_rack_type(df, base_rack_id, rack_templates, status)
                df_final = assign_sequential_location_ids(df_a)
                
                if not df_final.empty:
                    st.subheader("📊 Allocation Data")
                    ex_buf = io.BytesIO()
                    df_final.to_excel(ex_buf, index=False)
                    st.download_button("📥 Download Excel", ex_buf.getvalue(), "Allocation.xlsx")
                    
                    prog = st.progress(0)
                    if output_type == "Rack Labels":
                        pdf, _ = generate_rack_labels(df_final, prog)
                        st.download_button("📥 Download PDF", pdf, "Rack_Labels.pdf")
                    elif output_type == "Bin Labels":
                        mtm = [m.strip() for m in [m1, m2, m3] if m.strip()]
                        pdf, _ = generate_bin_labels(df_final, mtm, prog, status)
                        st.download_button("📥 Download PDF", pdf, "Bin_Labels.pdf")
                    elif output_type == "Rack List":
                        pdf, _ = generate_rack_list_pdf(df_final, base_rack_id, None, prog)
                        st.download_button("📥 Download PDF", pdf, "Rack_List.pdf")
                    prog.empty(); status.empty()
        else:
            st.error("❌ Missing required columns.")

if __name__ == "__main__":
    main()
