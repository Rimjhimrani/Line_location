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

# --- EXACT STYLE DEFINITIONS FROM YOUR CODE ---
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32, wordWrap='CJK')
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16)

bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=14, alignment=TA_LEFT)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica-Bold', fontSize=12, alignment=TA_CENTER)

rl_header_style = ParagraphStyle(name='RLHeader', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)
location_header_style = ParagraphStyle(name='LocHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER)

# --- HELPER FUNCTIONS ---

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return (0, 0)
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b>", bold_style_v2)

def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(version=1, box_size=10, border=1)
    qr.add_data(data_string)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return RLImage(buf, width=2.5*cm, height=2.5*cm)

def find_required_columns(df):
    cols = {col.upper().strip(): col for col in df.columns}
    return {
        'Part No': cols.get(next((k for k in cols if 'PART' in k and ('NO' in k or 'NUM' in k)), None)),
        'Description': cols.get(next((k for k in cols if 'DESC' in k), None)),
        'Bus Model': cols.get(next((k for k in cols if 'BUS' in k and 'MODEL' in k), None)),
        'Station No': cols.get(next((k for k in cols if 'STATION' in k), None)),
        'Container': cols.get(next((k for k in cols if 'CONTAINER' in k), None)),
        'Qty/Bin': cols.get(next((k for k in cols if 'BIN' in k and 'QTY' in k), None)),
        'Qty/Veh': cols.get(next((k for k in cols if 'VEH' in k and 'QTY' in k), None))
    }

def extract_location_values(row):
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]

def extract_store_location_data_from_excel(row):
    return [str(row.get(c, '')) for c in ['Storage', 'Area', 'Zone', 'Row', 'Rack_S', 'Level_S', 'Bin_S']]

# --- DYNAMIC ASSIGNMENT LOGIC (STATION-WISE RESET) ---

def automate_station_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    cols = find_required_columns(df)
    df_p = df.copy()
    rename_map = {cols[k]: k for k in cols if cols[k]}
    df_p.rename(columns=rename_map, inplace=True)
    
    df_p['bin_area'] = df_p['Container'].map(lambda x: bin_info_map.get(x, {}).get('dims', (0,0))[0] * bin_info_map.get(x, {}).get('dims', (0,0))[1])
    df_p['bins_per_cell'] = df_p['Container'].map(lambda x: bin_info_map.get(x, {}).get('capacity', 1))
    
    final_data = []
    for st_no, st_group in df_p.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Calculating for Station: {st_no}")
        
        container_groups = sorted(st_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        total_cells_needed = 0
        for _, c_df in container_groups:
            cap = c_df['bins_per_cell'].iloc[0]
            total_cells_needed += math.ceil(len(c_df) / (cap if cap > 0 else 1))
        
        cells_per_rack = len(levels) * cells_per_level
        num_racks = math.ceil(total_cells_needed / cells_per_rack)
        
        available_locs = []
        for r_idx in range(1, num_racks + 1):
            r_str = f"{r_idx:02d}"
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    available_locs.append({'Rack No 1st': r_str[0], 'Rack No 2nd': r_str[1], 'Level': lvl, 'Cell': str(c_idx), 'Rack': base_rack_id})

        ptr = 0
        for _, c_df in container_groups:
            parts = c_df.to_dict('records')
            cap = int(parts[0]['bins_per_cell'])
            for i in range(0, len(parts), cap):
                chunk = parts[i:i + cap]
                for p in chunk:
                    if ptr < len(available_locs):
                        p.update(available_locs[ptr])
                        final_data.append(p)
                ptr += 1
        
        for i in range(ptr, len(available_locs)):
            empty = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': st_group['Bus Model'].iloc[0], 'Station No': st_no, 'Container': '', 'Qty/Bin': '', 'Qty/Veh': ''}
            empty.update(available_locs[i])
            final_data.append(empty)
            
    return pd.DataFrame(final_data)

# --- EXACT PDF GENERATION FUNCTIONS (UNALTERED FORMAT) ---

def generate_bin_labels(df, mtm_models, progress_bar=None, status_text=None):
    if not QR_AVAILABLE:
        st.error("❌ QR Code library not found. Please install `qrcode` and `Pillow`.")
        return None, {}

    STICKER_WIDTH, STICKER_HEIGHT = 10 * cm, 15 * cm
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT),
                            topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm,
                            leftMargin=0.1*cm, rightMargin=0.1*cm)

    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_filtered.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    total_labels = len(df_filtered)
    label_summary = {}
    all_elements = []

    def draw_border(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(1.8)
        canvas.rect(0.1*cm, STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        if status_text: status_text.text(f"Processing Bin Label {i+1}/{total_labels}")
        
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        part_no, desc = str(row.get('Part No', '')), str(row.get('Description', ''))
        qty_bin, qty_veh = str(row.get('Qty/Bin', '')), str(row.get('Qty/Veh', ''))

        store_loc_raw = extract_store_location_data_from_excel(row)
        line_loc_raw = extract_location_values(row)

        qr_data = f"Part No: {part_no}\nDesc: {desc}\nQty/Bin: {qty_bin}\nLine Loc: {'|'.join(line_loc_raw)}"
        qr_image = generate_qr_code_image(qr_data)
        
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([
            ["Part No", Paragraph(f"{part_no}", bin_bold_style)],
            ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, bin_desc_style)],
            ["Qty/Bin", Paragraph(qty_bin, bin_qty_style)]
        ], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(0,-1), 'Helvetica'), ('FONTSIZE', (0,0),(0,-1), 11)]))

        inner_table_width = content_width * 2 / 3
        col_props = [1.8, 2.4, 0.7, 0.7, 0.7, 0.7, 0.9]
        inner_col_widths = [w * inner_table_width / sum(col_props) for w in col_props]
        
        store_loc_inner = Table([store_loc_raw], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        store_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        store_loc_table = Table([[Paragraph("Store Location", bin_desc_style), store_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        store_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        
        line_loc_inner = Table([line_loc_raw], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        line_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        line_loc_table = Table([[Paragraph("Line Location", bin_desc_style), line_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        line_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        if mtm_models:
            mtm_qty_values = [Paragraph(f"<b>{qty_veh}</b>", bin_qty_style) if str(row.get('Bus Model','')).strip().upper() == m.upper() else "" for m in mtm_models]
            mtm_table = Table([mtm_models, mtm_qty_values], colWidths=[3.6*cm/len(mtm_models)]*len(mtm_models), rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        else: mtm_table = None

        bottom_row = Table([[mtm_table or "", Spacer(1, 1*cm), qr_image or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, content_width - 7.1*cm], rowHeights=[2.5*cm])
        bottom_row.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        all_elements.extend([main_table, store_loc_table, line_loc_table, Spacer(1, 0.2*cm), bottom_row])
        if i < total_labels - 1: all_elements.append(PageBreak())

    doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    
    df_clean = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    grouped = df_clean.groupby(['Station No', 'Rack No 1st', 'Rack No 2nd'])
    total_groups = len(grouped)
    
    master_value_style_left = ParagraphStyle(name='MasterValLeft', fontName='Helvetica-Bold', fontSize=13, alignment=TA_LEFT)
    master_value_style_center = ParagraphStyle(name='MasterValCenter', fontName='Helvetica-Bold', fontSize=13, alignment=TA_CENTER)

    for i, ((st_no, r1, r2), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1) / total_groups) * 100))
        rack_key = f"{r1}{r2}"
        
        top_logo_img = ""
        if top_logo_file:
            top_logo_img = RLImage(io.BytesIO(top_logo_file.getvalue()), width=top_logo_w*cm, height=top_logo_h*cm)
        
        elements.append(Table([[Paragraph("Document Ref No.:", rl_header_style), "", top_logo_img]], colWidths=[5*cm, 17.5*cm, 5*cm]))
        
        master_data = [
            [Paragraph("STATION NAME", rl_header_style), Paragraph(str(group.iloc[0].get('Station Name', 'N/A')), master_value_style_left),
             Paragraph("STATION NO", rl_header_style), Paragraph(str(st_no), master_value_style_center)],
            [Paragraph("MODEL", rl_header_style), Paragraph(str(group.iloc[0]['Bus Model']), master_value_style_left),
             Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {rack_key}", master_value_style_center)]
        ]
        master_table = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        master_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,-1), colors.HexColor("#8EAADB")), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        elements.append(master_table)
        
        p_data = [["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]]
        for idx, row in enumerate(group.to_dict('records')):
            loc_str = f"{row['Bus Model']}-{st_no}-{base_rack_id}{rack_key}-{row['Level']}{row['Cell']}"
            p_data.append([str(idx+1), row['Part No'], Paragraph(row['Description'], rl_cell_left_style), row['Container'], row['Qty/Bin'], loc_str])
            
        p_table = Table(p_data, colWidths=[1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm])
        p_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#F4B084")), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        elements.append(p_table)
        
        # FOOTER
        left_content = [Paragraph(f"<i>Creation Date: {datetime.date.today().strftime('%d-%m-%Y')}</i>", rl_cell_left_style),
                        Paragraph("<b>Verified by:</b>", rl_header_style), Paragraph("Name: ________________", rl_cell_left_style)]
        footer_table = Table([[left_content, Paragraph("Designed by Agilomatrix", TA_RIGHT)]], colWidths=[20*cm, 7.7*cm])
        elements.append(footer_table)
        elements.append(PageBreak())
        
    doc.build(elements)
    buffer.seek(0)
    return buffer, total_groups

def generate_rack_labels(df, progress_bar=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    df_p = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    total_labels = len(df_p)
    label_summary = {}

    for i, part in enumerate(df_p.to_dict('records')):
        if progress_bar: progress_bar.progress(int((i / total_labels) * 100))
        rk = f"ST-{part['Station No']} / Rack {part['Rack No 1st']}{part['Rack No 2nd']}"
        label_summary[rk] = label_summary.get(rk, 0) + 1
        if i > 0 and i % 4 == 0: elements.append(PageBreak())
        
        pt = Table([['Part No', format_part_no_v2(str(part['Part No']))], ['Description', format_description(str(part['Description']))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        loc_v = extract_location_values(part)
        lt = Table([[Paragraph('Line Location', location_header_style)] + loc_v], colWidths=[4*cm]+[11*cm/7]*7, rowHeights=[1.2*cm])
        
        pt.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        loc_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        lt_style = [('GRID', (0,0),(-1,-1), 1, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]
        for j, color in enumerate(loc_colors): lt_style.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        lt.setStyle(TableStyle(lt_style))
        elements.extend([pt, Spacer(1, 0.3*cm), lt, Spacer(1, 0.2*cm)])

    doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

# --- MAIN UI ---

def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.sidebar.title("📄 Configuration")
    out_type = st.sidebar.selectbox("Output Type:", ["Rack Labels", "Bin Labels", "Rack List"])
    base_id = st.sidebar.text_input("Infrastructure ID", "R")

    top_logo = None
    if out_type == "Rack List":
        top_logo = st.sidebar.file_uploader("Upload Logo")

    mtm_models = []
    if out_type == "Bin Labels":
        m1 = st.sidebar.text_input("Model 1", "7M"); m2 = st.sidebar.text_input("Model 2", "9M"); m3 = st.sidebar.text_input("Model 3", "12M")
        mtm_models = [m.strip() for m in [m1, m2, m3] if m.strip()]

    file = st.file_uploader("Upload Excel", type=['xlsx', 'xls', 'csv'])
    if file:
        df = pd.read_excel(file) if file.name.endswith('x') else pd.read_csv(file)
        cols = find_required_columns(df)
        if cols['Station No'] and cols['Container']:
            st.sidebar.subheader("Rack Geometry")
            cell_dim = st.sidebar.text_input("Cell Dimensions (L x W)", "800x400")
            lvls = st.sidebar.multiselect("Levels", ['A','B','C','D','E','F','G','H'], ['A','B','C','D'])
            c_per_l = st.sidebar.number_input("Physical Cells per Level", 1, 50, 10)
            
            u_conts = sorted(df[cols['Container']].dropna().unique())
            bin_map = {}
            for c in u_conts:
                st.sidebar.markdown(f"**{c}**")
                d = st.sidebar.text_input(f"Dim", "600x400", key=f"d_{c}")
                cp = st.sidebar.number_input(f"Capacity", 1, 10, 1, key=f"c_{c}")
                bin_map[c] = {'dims': parse_dimensions(d), 'capacity': cp}

            if st.button("🚀 Generate PDF Studio"):
                status = st.empty(); prog = st.progress(0)
                df_assigned = automate_station_assignment(df, base_id, lvls, c_per_l, bin_map, status)
                
                if out_type == "Rack Labels":
                    buf, summ = generate_rack_labels(df_assigned, prog)
                elif out_type == "Bin Labels":
                    buf, summ = generate_bin_labels(df_assigned, mtm_models, prog, status)
                else:
                    buf, count = generate_rack_list_pdf(df_assigned, base_id, top_logo, 4, 1.5, "", prog, status)
                    summ = {"Total Racks": count}

                st.download_button("📥 Download PDF", buf.getvalue(), f"{out_type}.pdf", "application/pdf")
                st.table(pd.DataFrame(list(summ.items()), columns=['Rack / Station', 'Count']))
                prog.empty(); status.success("Done!")
        else: st.error("❌ Missing required columns.")

if __name__ == "__main__":
    main()
