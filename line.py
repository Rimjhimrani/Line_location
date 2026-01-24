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
st.set_page_config(page_title="AgiloSmartTag Studio", page_icon="🏷️", layout="wide")

# --- Style Definitions ---
bold_style_v2 = ParagraphStyle(name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=32, spaceBefore=0, spaceAfter=2, wordWrap='CJK')
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2)
bin_bold_style = ParagraphStyle(name='BinBold', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER, leading=20)
bin_desc_style = ParagraphStyle(name='BinDesc', fontName='Helvetica', fontSize=10, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='BinQty', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER, leading=16)
rl_header_style = ParagraphStyle(name='RLHeader', fontName='Helvetica-Bold', fontSize=11, alignment=TA_LEFT)
rl_cell_left_style = ParagraphStyle(name='RLCellLeft', fontName='Helvetica', fontSize=10, alignment=TA_LEFT)

# --- Formatting Functions ---
def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY': return Paragraph(f"<b><font size=34>EMPTY</font></b>", bold_style_v2)
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
    station_no_key = next((k for k in cols if 'STATION' in k), None)
    container_type_key = next((k for k in cols if 'CONTAINER' in k), None)
    return {'Part No': cols.get(part_no_key), 'Description': cols.get(desc_key), 'Bus Model': cols.get(bus_model_key), 'Station No': cols.get(station_no_key), 'Container': cols.get(container_type_key)}

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def parse_dimensions(dim_str):
    if not isinstance(dim_str, str) or not dim_str: return 0, 0
    nums = [int(n) for n in re.findall(r'\d+', dim_str)]
    return (nums[0], nums[1]) if len(nums) >= 2 else (0, 0)

def generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text=None):
    required_cols = find_required_columns(df)
    df_processed = df.copy()
    rename_dict = {required_cols['Part No']: 'Part No', required_cols['Description']: 'Description', required_cols['Bus Model']: 'Bus Model', required_cols['Station No']: 'Station No', required_cols['Container']: 'Container'}
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    df_processed['bin_info'] = df_processed['Container'].map(bin_info_map)
    df_processed['bin_area'] = df_processed['bin_info'].apply(lambda x: x['dims'][0] * x['dims'][1] if x else 0)
    df_processed['bins_per_cell'] = df_processed['bin_info'].apply(lambda x: x['capacity'] if x else 1)
    final_assigned_data = []
    for station_no, station_group in df_processed.groupby('Station No', sort=True):
        if status_text: status_text.text(f"Processing Station: {station_no}...")
        station_cells_needed = 0
        container_groups = sorted(station_group.groupby('Container'), key=lambda x: x[1]['bin_area'].iloc[0], reverse=True)
        for _, cont_df in container_groups:
            cap = cont_df['bins_per_cell'].iloc[0]
            station_cells_needed += math.ceil(len(cont_df) / cap)
        cells_per_rack = len(levels) * cells_per_level
        racks_for_station = math.ceil(station_cells_needed / cells_per_rack)
        station_available_cells = []
        for r_idx in range(1, racks_for_station + 1):
            rack_str = f"{r_idx:02d}"
            r1, r2 = rack_str[0], rack_str[1]
            for lvl in sorted(levels):
                for c_idx in range(1, cells_per_level + 1):
                    station_available_cells.append({'Rack No 1st': r1, 'Rack No 2nd': r2, 'Level': lvl, 'Physical_Cell': f"{c_idx:02d}", 'Rack': base_rack_id})
        current_cell_ptr = 0
        for _, cont_df in container_groups:
            parts = cont_df.to_dict('records')
            cap = parts[0]['bins_per_cell']
            for i in range(0, len(parts), cap):
                chunk = parts[i:i + cap]
                loc = station_available_cells[current_cell_ptr]
                for p in chunk:
                    p.update(loc)
                    final_assigned_data.append(p)
                current_cell_ptr += 1
        for i in range(current_cell_ptr, len(station_available_cells)):
            empty_label = {'Part No': 'EMPTY', 'Description': '', 'Bus Model': station_group['Bus Model'].iloc[0], 'Station No': station_no, 'Container': ''}
            empty_label.update(station_available_cells[i])
            final_assigned_data.append(empty_label)
    return pd.DataFrame(final_assigned_data)

def generate_rack_type_assignment(df, base_rack_id, rack_configs, container_dims, status_text=None):
    required_cols = find_required_columns(df)
    df_processed = df.copy()
    rename_dict = {required_cols['Part No']: 'Part No', required_cols['Description']: 'Description', required_cols['Bus Model']: 'Bus Model', required_cols['Station No']: 'Station No', required_cols['Container']: 'Container'}
    df_processed.rename(columns={k: v for k, v in rename_dict.items() if k}, inplace=True)
    final_parts_list = []
    for station_no, station_group in df_processed.groupby('Station No', sort=False):
        if status_text: status_text.text(f"Processing station: {station_no}...")
        rack_idx, level_idx = 0, 0
        sorted_racks = sorted(rack_configs.items())
        for container_type, parts_group in station_group.groupby('Container', sort=True):
            items_to_place = parts_group.to_dict('records')
            while items_to_place:
                slot_found = False
                search_rack_idx, search_level_idx = rack_idx, level_idx
                while search_rack_idx < len(sorted_racks):
                    rack_name, config = sorted_racks[search_rack_idx]
                    levels, capacity = config.get('levels', []), config.get('rack_bin_counts', {}).get(container_type, 0)
                    if capacity > 0 and search_level_idx < len(levels):
                        slot_found = True
                        rack_idx, level_idx = search_rack_idx, search_level_idx
                        break
                    search_level_idx = 0
                    search_rack_idx += 1
                if not slot_found:
                    st.warning(f"⚠️ Ran out of rack space at Station {station_no} for '{container_type}'.")
                    break
                rack_name, config = sorted_racks[rack_idx]
                levels = config.get('levels', [])
                level_capacity = config.get('rack_bin_counts', {}).get(container_type, 0)
                parts_for_level = items_to_place[:level_capacity]
                items_to_place = items_to_place[level_capacity:]
                num_empty_slots = level_capacity - len(parts_for_level)
                level_items = parts_for_level + ([{'Part No': 'EMPTY'}] * num_empty_slots)
                item_template = {col: '' for col in df_processed.columns}
                for cell_idx, item in enumerate(level_items, 1):
                    rack_num_val = ''.join(filter(str.isdigit, rack_name))
                    rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
                    rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else rack_num_val[0]
                    location_info = {'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd, 'Level': levels[level_idx], 'Cell': str(cell_idx), 'Station No': station_no, 'Rack Key': f"{rack_num_1st}{rack_num_2nd}"}
                    if item['Part No'] == 'EMPTY':
                        full_item = item_template.copy()
                        full_item.update({'Part No': 'EMPTY', 'Container': container_type})
                    else:
                        full_item = item
                    full_item.update(location_info)
                    final_parts_list.append(full_item)
                level_idx += 1
                if level_idx >= len(levels):
                    level_idx = 0
                    rack_idx += 1
    if not final_parts_list: return pd.DataFrame()
    return pd.DataFrame(final_parts_list)

def assign_sequential_location_ids(df):
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
            if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']: return str(val).strip()
        return default
    return [get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'ST. NAME', 'Station Name Short', 'Station Name']), get_clean_value(['Store Location', 'STORELOCATION', 'Store_Location']), get_clean_value(['ABB ZONE', 'ABB_ZONE', 'ABBZONE', 'Zone']), get_clean_value(['ABB LOCATION', 'ABB_LOCATION', 'ABBLOCATION']), get_clean_value(['ABB FLOOR', 'ABB_FLOOR', 'ABBFLOOR']), get_clean_value(['ABB RACK NO', 'ABB_RACK_NO', 'ABBRACKNO']), get_clean_value(['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK', 'ABBLEVELINRACK'])]

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
    CONTENT_BOX_WIDTH, CONTENT_BOX_HEIGHT = 10 * cm, 7.2 * cm
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(STICKER_WIDTH, STICKER_HEIGHT), topMargin=0.2*cm, bottomMargin=STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, leftMargin=0.1*cm, rightMargin=0.1*cm)
    df_filtered = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df_filtered.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    total_labels = len(df_filtered)
    label_summary = {}
    all_elements = []
    def draw_border(canvas, doc):
        canvas.saveState()
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(1.8)
        canvas.rect(x_offset + doc.leftMargin, y_offset, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()
    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        if status_text: status_text.text(f"Processing Bin Label {i+1}/{total_labels}")
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
        part_no = str(row.get('Part No', ''))
        desc = str(row.get('Description', ''))
        qty_bin = str(row.get('Qty/Bin', ''))
        qty_veh = str(row.get('Qty/Veh', ''))
        store_loc_raw = extract_store_location_data_from_excel(row)
        line_loc_raw = extract_location_values(row)
        qr_data = f"Part No: {part_no}\nDesc: {desc}\nQty/Bin: {qty_bin}\nQty/Veh: {qty_veh}\nStore Loc: {'|'.join([str(x).strip() for x in store_loc_raw])}\nLine Loc: {'|'.join([str(x).strip() for x in line_loc_raw])}"
        qr_image = generate_qr_code_image(qr_data)
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([["Part No", Paragraph(f"{part_no}", bin_bold_style)], ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, bin_desc_style)], ["Qty/Bin", Paragraph(qty_bin, bin_qty_style)]], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(0,-1), 'Helvetica'), ('FONTSIZE', (0,0),(0,-1), 11)]))
        inner_table_width = content_width * 2 / 3
        col_props = [1.8, 2.4, 0.7, 0.7, 0.7, 0.7, 0.9]
        inner_col_widths = [w * inner_table_width / sum(col_props) for w in col_props]
        store_loc_values = extract_store_location_data_from_excel(row)
        store_loc_inner = Table([store_loc_values], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        store_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        store_loc_table = Table([[Paragraph("Store Location", bin_desc_style), store_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        store_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        line_loc_values = extract_location_values(row)
        line_loc_inner = Table([line_loc_values], colWidths=inner_col_widths, rowHeights=[0.5*cm])
        line_loc_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        line_loc_table = Table([[Paragraph("Line Location", bin_desc_style), line_loc_inner]], colWidths=[content_width/3, inner_table_width], rowHeights=[0.5*cm])
        line_loc_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        mtm_table = None
        if mtm_models:
            qty_veh = str(row.get('Qty/Veh', ''))
            bus_model = str(row.get('Bus Model', '')).strip().upper()
            mtm_qty_values = []
            for model in mtm_models:
                if bus_model == model.strip().upper() and qty_veh:
                    mtm_qty_values.append(Paragraph(f"<b>{qty_veh}</b>", bin_qty_style))
                else:
                    mtm_qty_values.append("")
            mtm_data = [mtm_models, mtm_qty_values]
            num_models = len(mtm_models)
            total_mtm_width = 3.6 * cm
            col_width = total_mtm_width / num_models if num_models > 0 else total_mtm_width
            mtm_table = Table(mtm_data, colWidths=[col_width] * num_models, rowHeights=[0.75*cm, 0.75*cm])
            mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE'), ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0),(-1,-1), 9)]))
        mtm_width, qr_width, gap_width = 3.6 * cm, 2.5 * cm, 1.0 * cm
        remaining_width = content_width - mtm_width - gap_width - qr_width
        bottom_row_content = [mtm_table if mtm_table else "", "", qr_image or "", ""]
        bottom_row = Table([bottom_row_content], colWidths=[mtm_width, gap_width, qr_width, remaining_width], rowHeights=[2.5*cm])
        bottom_row.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))
        all_elements.extend([main_table, store_loc_table, line_loc_table, Spacer(1, 0.2*cm), bottom_row])
        if i < total_labels - 1: all_elements.append(PageBreak())
    if all_elements: doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
    df['Rack Key'] = df.apply(lambda x: f"{x.get('Rack No 1st', '')}{x.get('Rack No 2nd', '')}", axis=1)
    df.sort_values(by=['Station No', 'Rack Key', 'Level', 'Cell'], inplace=True)
    grouped = df.groupby(['Station No', 'Rack Key'])
    total_groups = len(grouped)
    has_zone = 'Zone' in df.columns and df['Zone'].notna().any()
    master_value_style_left = ParagraphStyle(name='MasterValLeft', fontName='Helvetica-Bold', fontSize=13, alignment=TA_LEFT)
    master_value_style_center = ParagraphStyle(name='MasterValCenter', fontName='Helvetica-Bold', fontSize=13, alignment=TA_CENTER)
    for i, ((station_no, rack_key), group) in enumerate(grouped):
        if progress_bar: progress_bar.progress(int(((i+1) / total_groups) * 100))
        if status_text: status_text.text(f"Generating List for Station {station_no} / Rack {rack_key}")
        first_row = group.iloc[0]
        station_name = str(first_row.get('Station Name', ''))
        bus_model = str(first_row.get('Bus Model', ''))
        top_logo_img = ""
        if top_logo_file:
            try:
                img_io = io.BytesIO(top_logo_file.getvalue())
                top_logo_img = RLImage(img_io, width=top_logo_w*cm, height=top_logo_h*cm)
            except: pass
        header_table = Table([[Paragraph("Document Ref No.:", rl_header_style), "", top_logo_img]], colWidths=[5*cm, 17.5*cm, 5*cm])
        header_table.setStyle(TableStyle([('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        master_label = Paragraph("<b>MASTER PARTS LIST</b>", master_value_style_center)
        master_data = [
            ["Station Name:", Paragraph(station_name, master_value_style_left), "Station No:", Paragraph(str(station_no), master_value_style_center), "Bus Model:", Paragraph(bus_model, master_value_style_center)],
            ["", "", "Rack ID:", Paragraph(base_rack_id, master_value_style_center), "Rack No:", Paragraph(rack_key, master_value_style_center)]
        ]
        
        if has_zone:
            zone_value = str(first_row.get('Zone', ''))
            master_data[1].extend(["Zone:", Paragraph(zone_value, master_value_style_center)])
            master_col_widths = [3.5*cm, 6*cm, 2.5*cm, 2*cm, 2.5*cm, 2*cm, 2*cm, 2*cm]
        else:
            master_col_widths = [3.5*cm, 6*cm, 2.5*cm, 2*cm, 2.5*cm, 2*cm]
        
        master_table = Table(master_data, colWidths=master_col_widths, rowHeights=[0.8*cm, 0.8*cm])
        master_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10)
        ]))

        parts_header = ["Sr. No.", "Part No", "Description", "Level", "Cell", "Qty/Bin", "Qty/Veh"]
        parts_data = [parts_header]
        
        for idx, (_, row) in enumerate(group.iterrows(), 1):
            part_no = str(row.get('Part No', ''))
            desc = str(row.get('Description', ''))
            level = str(row.get('Level', ''))
            cell = str(row.get('Cell', ''))
            qty_bin = str(row.get('Qty/Bin', ''))
            qty_veh = str(row.get('Qty/Veh', ''))
            
            parts_data.append([
                Paragraph(str(idx), rl_cell_left_style),
                Paragraph(part_no, rl_cell_left_style),
                Paragraph(desc, rl_cell_left_style),
                Paragraph(level, rl_cell_left_style),
                Paragraph(cell, rl_cell_left_style),
                Paragraph(qty_bin, rl_cell_left_style),
                Paragraph(qty_veh, rl_cell_left_style)
            ])
        
        parts_table = Table(parts_data, colWidths=[1.5*cm, 4*cm, 10*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm], repeatRows=1)
        parts_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 9)
        ]))

        footer_table_data = [["", ""]]
        if fixed_logo_path and os.path.exists(fixed_logo_path):
            try:
                fixed_logo_img = RLImage(fixed_logo_path, width=4*cm, height=1.5*cm)
                footer_table_data = [[fixed_logo_img, Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", rl_cell_left_style)]]
            except:
                footer_table_data = [["", Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", rl_cell_left_style)]]
        else:
            footer_table_data = [["", Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", rl_cell_left_style)]]
        
        footer_table = Table(footer_table_data, colWidths=[4*cm, 20.5*cm])
        footer_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))

        elements.extend([header_table, Spacer(1, 0.3*cm), master_label, Spacer(1, 0.2*cm), master_table, Spacer(1, 0.3*cm), parts_table, Spacer(1, 0.3*cm), footer_table])
        
        if i < total_groups - 1:
            elements.append(PageBreak())
    
    if elements:
        doc.build(elements)
    
    buffer.seek(0)
    return buffer

# --- Streamlit App ---
st.title("🏷️ AgiloSmartTag Studio")
st.markdown("**Generate professional rack and bin labels with location tracking**")

uploaded_file = st.file_uploader("📂 Upload Excel File", type=['xlsx', 'xls'])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ File loaded successfully! ({len(df)} rows)")
        
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
        
        required_cols = find_required_columns(df)
        if not all([required_cols['Part No'], required_cols['Description'], required_cols['Station No']]):
            st.error("❌ Missing required columns: Part No, Description, Station No")
            st.stop()
        
        st.sidebar.header("⚙️ Configuration")
        
        assignment_mode = st.sidebar.selectbox("Assignment Mode", ["Station-Wise", "Rack-Type"])
        base_rack_id = st.sidebar.text_input("Base Rack ID", "R")
        
        if assignment_mode == "Station-Wise":
            levels_input = st.sidebar.text_input("Levels (comma-separated)", "A,B,C,D")
            levels = [l.strip() for l in levels_input.split(",") if l.strip()]
            cells_per_level = st.sidebar.number_input("Cells per Level", min_value=1, value=10)
            
            containers = get_unique_containers(df, required_cols['Container'])
            if not containers:
                st.error("❌ No container types found")
                st.stop()
            
            st.sidebar.subheader("Container Configuration")
            bin_info_map = {}
            for cont in containers:
                with st.sidebar.expander(f"📦 {cont}"):
                    dims_str = st.text_input(f"Dimensions (WxH)", "200x150", key=f"dim_{cont}")
                    capacity = st.number_input(f"Bins per Cell", min_value=1, value=1, key=f"cap_{cont}")
                    dims = parse_dimensions(dims_str)
                    bin_info_map[cont] = {'dims': dims, 'capacity': capacity}
            
            if st.button("🚀 Generate Labels"):
                with st.spinner("Processing..."):
                    status_text = st.empty()
                    df_assigned = generate_station_wise_assignment(df, base_rack_id, levels, cells_per_level, bin_info_map, status_text)
                    df_final = assign_sequential_location_ids(df_assigned)
                    
                    st.success("✅ Assignment complete!")
                    st.dataframe(df_final)
                    
                    progress_bar = st.progress(0)
                    rack_pdf, rack_summary = generate_rack_labels(df_final, progress_bar)
                    
                    st.download_button("📥 Download Rack Labels", rack_pdf, "rack_labels.pdf", "application/pdf")
                    st.dataframe(rack_summary)
        
        else:  # Rack-Type mode
            st.sidebar.subheader("Rack Configuration")
            num_racks = st.sidebar.number_input("Number of Racks", min_value=1, value=2)
            
            rack_configs = {}
            for i in range(1, num_racks + 1):
                with st.sidebar.expander(f"Rack {i:02d}"):
                    levels_input = st.text_input(f"Levels", "A,B,C,D", key=f"rack_{i}_levels")
                    levels = [l.strip() for l in levels_input.split(",") if l.strip()]
                    
                    containers = get_unique_containers(df, required_cols['Container'])
                    rack_bin_counts = {}
                    for cont in containers:
                        count = st.number_input(f"{cont} bins per level", min_value=0, value=0, key=f"rack_{i}_{cont}")
                        rack_bin_counts[cont] = count
                    
                    rack_configs[f"Rack{i:02d}"] = {'levels': levels, 'rack_bin_counts': rack_bin_counts}
            
            container_dims = {}
            containers = get_unique_containers(df, required_cols['Container'])
            for cont in containers:
                dims_str = st.sidebar.text_input(f"{cont} Dimensions", "200x150", key=f"dim_rt_{cont}")
                container_dims[cont] = parse_dimensions(dims_str)
            
            if st.button("🚀 Generate Labels (Rack-Type)"):
                with st.spinner("Processing..."):
                    status_text = st.empty()
                    df_final = generate_rack_type_assignment(df, base_rack_id, rack_configs, container_dims, status_text)
                    
                    if not df_final.empty:
                        st.success("✅ Assignment complete!")
                        st.dataframe(df_final)
                        
                        progress_bar = st.progress(0)
                        rack_pdf, rack_summary = generate_rack_labels(df_final, progress_bar)
                        
                        st.download_button("📥 Download Rack Labels", rack_pdf, "rack_labels.pdf", "application/pdf")
                        st.dataframe(rack_summary)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
