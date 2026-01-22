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
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# Try to import QR code libraries
try:
    import qrcode
    from PIL import Image
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- Style Definitions ---
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32, wordWrap='CJK')
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)

# Bin Label Specific Styles
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=14, alignment=TA_LEFT)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica-Bold', fontSize=12, alignment=TA_CENTER)

# Rack List Specific Styles
rl_header_style = ParagraphStyle(name='RLHeader', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)
rl_cell_center_style = ParagraphStyle(name='RLCellCenter', fontName='Helvetica', fontSize=10, alignment=TA_CENTER)

# --- Helper Functions ---
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

def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    return {
        'Part No': cols.get(next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), '')),
        'Description': cols.get(next((k for k in cols if 'DESC' in k), '')),
        'Bus Model': cols.get(next((k for k in cols if 'BUS' in k and 'MODEL' in k), '')),
        'Station No': cols.get(next((k for k in cols if 'STATION' in k), '')),
        'Container': cols.get(next((k for k in cols if 'CONTAINER' in k), '')),
        'Qty/Bin': cols.get(next((k for k in cols if 'BIN' in k and 'QTY' in k), '')),
        'Qty/Veh': cols.get(next((k for k in cols if 'VEH' in k and 'QTY' in k), ''))
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def generate_qr_code_image(data):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(version=1, box_size=10, border=1)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return RLImage(img_byte_arr, width=2.5*cm, height=2.5*cm)

def extract_store_location_data_from_excel(row):
    # Looking for storage columns (Storage, Area, Row, Level, Bin)
    # If not found, returns empty strings for the 7 slots expected by the template
    return [str(row.get(c, '')) for c in ['Storage', 'Area', 'Row', 'Rack', 'Level', 'Bin', 'Side']]

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

# --- Core logic for Rack Assignment (Station-wise reset) ---
def generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_processed = df.copy()
    rename_dict = {cols['Part No']: 'Part No', cols['Description']: 'Description', cols['Bus Model']: 'Bus Model', 
                   cols['Station No']: 'Station No', cols['Container']: 'Container', cols['Qty/Bin']: 'Qty/Bin', cols['Qty/Veh']: 'Qty/Veh'}
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    
    df_processed['bin_area'] = df_processed['Container'].map(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
    df_processed['bins_per_cell'] = df_processed['Container'].map(lambda x: bin_info_map.get(x, {}).get('capacity', 1))
    
    final_assigned_data = []
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Calculating for Station: {station_no}")
        station_cells_needed = 0
        container_groups = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        for _, cont_df in container_groups:
            cap = cont_df['bins_per_cell'].iloc[0]
            station_cells_needed += math.ceil(len(cont_df) / (cap if cap > 0 else 1))
        
        cells_per_rack = len(levels) * cells_per_level
        racks_for_station = math.ceil(station_cells_needed / cells_per_rack)
        
        station_cells = []
        for r_idx in range(1, racks_for_station + 1):
            r_str = f"{r_idx:02d}"
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    station_cells.append({'Rack No 1st': r_str[0], 'Rack No 2nd': r_str[1], 'Level': lvl, 'Physical_Cell': f"{c_idx:02d}", 'Rack': base_rack_id})

        ptr = 0
        for _, cont_df in container_groups:
            parts = cont_df.to_dict('records')
            cap = int(parts[0]['bins_per_cell'])
            for i in range(0, len(parts), cap):
                chunk = parts[i:i + cap]
                for p in chunk:
                    p.update(station_cells[ptr])
                    final_assigned_data.append(p)
                ptr += 1
        
        for i in range(ptr, len(station_cells)):
            empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': station_group['Bus Model'].iloc[0], 'Station No': station_no, 'Container': '', 'Qty/Bin': '', 'Qty/Veh': ''}
            empty.update(station_cells[i])
            final_assigned_data.append(empty)

    return pd.DataFrame(final_assigned_data)

# --- Bin Label Generation (Incorporated) ---
def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE:
        st.error("❌ qrcode/Pillow not found.")
        return None, {}

    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)

    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total_labels = len(df_filtered)
    label_summary = {}
    elements = []

    def draw_border(canvas, doc):
        canvas.saveState()
        canvas.rect(0.1*cm, STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1)/total_labels)*100))
        rack_key = f"ST-{row.get('Station No','')} / Rack {row.get('Rack No 1st','')}{row.get('Rack No 2nd','')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        p_no, dsc = str(row.get('Part No','')), str(row.get('Description',''))
        q_bin, q_veh = str(row.get('Qty/Bin','')), str(row.get('Qty/Veh',''))
        
        qr_data = f"Part No: {p_no}\nDesc: {dsc}\nQty/Bin: {q_bin}\nLoc: {row.get('Rack','')}{row.get('Rack No 1st','')}{row.get('Rack No 2nd','')}-{row.get('Level','')}{row.get('Physical_Cell','')}"
        qr_img = generate_qr_code_image(qr_data)

        content_w = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([["Part No", Paragraph(p_no, bin_bold_style)], ["Description", Paragraph(dsc[:47], bin_desc_style)], ["Qty/Bin", Paragraph(q_bin, bin_qty_style)]], colWidths=[content_w/3, content_w*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # Location rows
        store_vals = extract_store_location_data_from_excel(row)
        line_vals = extract_location_values(row)
        l_inner = Table([line_vals], colWidths=[content_w*2/3/7]*7, rowHeights=[0.5*cm])
        l_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('FONTSIZE', (0,0),(-1,-1), 8), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        line_table = Table([[Paragraph("Line Location", bin_desc_style), l_inner]], colWidths=[content_w/3, content_w*2/3])
        
        # MTM Section
        mtm_data = [mtm_models, [Paragraph(f"<b>{q_veh}</b>", bin_qty_style) if str(row.get('Bus Model','')).strip().upper() == m.upper() else "" for m in mtm_models]]
        mtm_table = Table(mtm_data, colWidths=[3.6*cm/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm]*2)
        mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        bottom_row = Table([[mtm_table, Spacer(1, 1*cm), qr_img if qr_img else ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm], rowHeights=[2.5*cm])
        
        elements.extend([main_table, line_table, Spacer(1, 0.2*cm), bottom_row, PageBreak()])

    doc.build(elements, onFirstPage=draw_border, onLaterPages=draw_border)
    return buffer, label_summary

# --- Rack List Generation (Incorporated) ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    
    df_clean = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    grouped = df_clean.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd'])

    for i, ((st_no, r1, r2), group) in enumerate(grouped):
        rack_key = f"{r1}{r2}"
        if progress_bar: progress_bar.progress(int(((i+1)/len(grouped))*100))
        
        # Header with Logo
        top_img = ""
        if top_logo_file:
            top_img = RLImage(io.BytesIO(top_logo_file.getvalue()), width=top_logo_w*cm, height=top_logo_h*cm)
        
        elements.append(Table([[Paragraph("Rack Inventory List", rl_header_style), "", top_img]], colWidths=[5*cm, 17.5*cm, 5*cm]))
        
        # Master Info
        m_data = [["STATION NO", str(st_no), "RACK NO", f"Rack - {rack_key}"], ["MODEL", str(group.iloc[0]['Bus Model']), "", ""]]
        m_table = Table(m_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm]*2)
        m_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#8EAADB"))]))
        elements.append(m_table)
        
        # Parts Table
        p_data = [["S.NO", "PART NO", "DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, r in enumerate(group.to_dict('records')):
            loc = f"{r['Bus Model']}-{st_no}-{base_rack_id}{rack_key}-{r['Level']}{r['Physical_Cell']}"
            p_data.append([idx+1, r['Part No'], Paragraph(r['Description'], rl_cell_left_style), r['Container'], r['Qty/Bin'], loc])
        
        p_table = Table(p_data, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        elements.append(p_table)
        elements.append(PageBreak())

    doc.build(elements)
    return buffer, len(grouped)

# --- Rack Label Generation (From previous version) ---
def generate_rack_labels(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df_p = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    label_summary = {}

    for i, part in enumerate(df_p.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i/len(df_p))*100))
        rk = f"ST-{part['Station No']} / Rack {part['Rack No 1st']}{part['Rack No 2nd']}"
        label_summary[rk] = label_summary.get(rk, 0) + 1
        
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        pt = Table([['Part No', format_part_no_v2(str(part['Part No']))], ['Description', format_description(str(part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_v = extract_location_values(part)
        # Using placeholder 'Cell' as Physical Cell for simple rack labels
        loc_v[-1] = part['Physical_Cell'] 
        lt = Table([['Line Location'] + loc_v], colWidths=[4*cm]+[11*cm/7]*7, rowHeights=[1.2*cm])
        
        pt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        lt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('BACKGROUND', (1,0),(-1,0), colors.lightblue)]))
        
        elements.extend([pt, Spacer(1, 0.3*cm), lt, Spacer(1, 0.2*cm)])

    doc.build(elements)
    return buffer, label_summary

# --- Main UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("Designed by Agilomatrix | Station-wise Rack Auto-Generation")
    
    st.sidebar.title("📄 Configuration")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    base_id = st.sidebar.text_input("Infrastructure ID (e.g. R, TR)", "R")

    top_logo = None
    if output_type == "Rack List":
        top_logo = st.sidebar.file_uploader("Upload Company Logo")

    mtm_models = []
    if output_type == "Bin Labels":
        m1 = st.sidebar.text_input("Model 1", "7M")
        m2 = st.sidebar.text_input("Model 2", "9M")
        m3 = st.sidebar.text_input("Model 3", "12M")
        mtm_models = [m for m in [m1, m2, m3] if m]

    file = st.file_uploader("Upload Excel", type=['xlsx', 'xls', 'csv'])
    if file:
        df = pd.read_excel(file) if file.name.endswith('x') else pd.read_csv(file)
        cols = find_required_columns(df)
        
        if cols['Station No'] and cols['Container']:
            st.sidebar.subheader("Rack Geometry")
            lvls = st.sidebar.multiselect("Levels", ['A','B','C','D','E','F'], ['A','B','C','D'])
            c_per_l = st.sidebar.number_input("Cells per Level", 1, 50, 10)
            
            u_conts = get_unique_containers(df, cols['Container'])
            bin_map = {}
            for c in u_conts:
                st.sidebar.markdown(f"**{c}**")
                d = st.sidebar.text_input(f"Dim", "600x400", key=f"d_{c}")
                cp = st.sidebar.number_input(f"Cap", 1, 10, 1, key=f"c_{c}")
                bin_map[c] = {'dims': parse_dimensions(d), 'capacity': cp}

            if st.button("🚀 Generate PDF"):
                status = st.empty()
                df_assigned = generate_station_wise_assignment(df, base_id, lvls, c_per_l, bin_map, status)
                
                prog = st.progress(0)
                if output_type == "Rack Labels":
                    buf, summary = generate_rack_labels(df_assigned, prog)
                elif output_type == "Bin Labels":
                    buf, summary = generate_bin_labels(df_assigned, mtm_models, prog, status)
                else:
                    buf, count = generate_rack_list_pdf(df_assigned, base_id, top_logo, 4, 1.5, "", prog)
                    summary = {"Total Racks": count}

                st.download_button("📥 Download PDF", buf.getvalue(), f"{output_type}.pdf", "application/pdf")
                st.table(pd.DataFrame(list(summary.items()), columns=['Key', 'Value']))
                prog.empty()
        else:
            st.error("Missing Station or Container columns.")

if __name__ == "__main__":
    main()
