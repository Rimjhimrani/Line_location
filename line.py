import streamlit as st
import pandas as pd
import os
import io
import re
import datetime
from io import BytesIO

# --- ReportLab Imports ---
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image as RLImage
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# --- Dependency Check for Bin Labels (QR Codes) ---
try:
    import qrcode
    from PIL import Image as PILImage
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False


# --- Page Configuration ---
st.set_page_config(
    page_title="AgiloSmartTag Studio",
    page_icon="🏷️",
    layout="wide"
)

# --- Style Definitions (Shared) ---
bold_style_v1 = ParagraphStyle(
    name='Bold_v1', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=5, spaceAfter=2
)
bold_style_v2 = ParagraphStyle(
    name='Bold_v2', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT, leading=16, spaceBefore=10, spaceAfter=15,
)
desc_style = ParagraphStyle(
    name='Description', fontName='Helvetica', fontSize=20, alignment=TA_LEFT, leading=16, spaceBefore=2, spaceAfter=2
)
location_header_style = ParagraphStyle(
    name='LocationHeader', fontName='Helvetica', fontSize=16, alignment=TA_CENTER, leading=18
)

# Base styles for location (will be overridden by dynamic logic)
location_value_style_base = ParagraphStyle(
    name='LocationValue_Base', fontName='Helvetica', alignment=TA_CENTER
)

# --- Style Definitions (Bin-Label Specific) ---
bin_bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, leading=14)
bin_desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
bin_qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

# --- Style Definitions (Rack List Specific) ---
rl_header_style = ParagraphStyle(name='RL_Header', fontName='Helvetica-Bold', fontSize=10, alignment=TA_LEFT)
rl_cell_style = ParagraphStyle(name='RL_Cell', fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
rl_cell_left_style = ParagraphStyle(name='RL_Cell_Left', fontName='Helvetica', fontSize=9, alignment=TA_LEFT)


# --- Formatting Functions ---
def format_part_no_v1(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=17>{part1}</font><font size=22>{part2}</font></b>", bold_style_v1)
    return Paragraph(f"<b><font size=17>{part_no}</font></b>", bold_style_v1)

def format_part_no_v2(part_no):
    if not part_no or not isinstance(part_no, str): part_no = str(part_no)
    if part_no.upper() == 'EMPTY':
         return Paragraph(f"<b><font size=34>EMPTY</font></b><br/><br/>", bold_style_v2)
    if len(part_no) > 5:
        part1, part2 = part_no[:-5], part_no[-5:]
        return Paragraph(f"<b><font size=34>{part1}</font><font size=40>{part2}</font></b><br/><br/>", bold_style_v2)
    return Paragraph(f"<b><font size=34>{part_no}</font></b><br/><br/>", bold_style_v2)

def format_description_v1(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    font_size = 15 if len(desc) <= 30 else 13 if len(desc) <= 50 else 11 if len(desc) <= 70 else 10 if len(desc) <= 90 else 9
    desc_style_v1 = ParagraphStyle(name='Description_v1', fontName='Helvetica', fontSize=font_size, alignment=TA_LEFT, leading=font_size + 2)
    return Paragraph(desc, desc_style_v1)

def format_description(desc):
    if not desc or not isinstance(desc, str): desc = str(desc)
    return Paragraph(desc, desc_style)

# --- NEW FUNCTION: Dynamic Autosizing for Location Cells ---
def get_dynamic_location_style(text, column_type):
    """
    Returns a ParagraphStyle with font size adjusted based on text length 
    and specific column type (Bus Model vs Station No vs Others).
    """
    text_len = len(str(text))
    font_name = 'Helvetica'
    
    # Defaults
    font_size = 16
    leading = 18

    if column_type == 'Bus Model':
        # Logic for Bus Model (Index 0)
        if text_len <= 3: 
            font_size = 14; leading = 18
        elif text_len <= 5: 
            font_size = 12; leading = 18
        elif text_len <= 10: 
            font_size = 10; leading = 15
        else: 
            font_size = 11; leading = 13
            
    elif column_type == 'Station No':
        # Logic for Station No (Index 1) - Often needs to fit "ST-01" etc.
        if text_len <= 2: 
            font_size = 20; leading = 21
        elif text_len <= 5: 
            font_size = 18; leading = 21
        elif text_len <= 8: 
            font_size = 15; leading = 15
        else: 
            font_size = 11; leading = 13
            
    else:
        # Logic for standard cells (Rack, Level, Cell) - usually short
        if text_len <= 2: 
            font_size = 16; leading = 18
        elif text_len <= 4:
            font_size = 14; leading = 18
        else:
            font_size = 12; leading = 14

    return ParagraphStyle(
        name=f'Dyn_{column_type}_{text_len}',
        parent=location_value_style_base,
        fontName=font_name,
        fontSize=font_size,
        leading=leading,
        alignment=TA_CENTER
    )


# --- Core Logic Functions ---
def find_required_columns(df):
    cols_map = {col.strip().upper(): col for col in df.columns}
    
    def find_col(patterns):
        for p in patterns:
            if p in cols_map:
                return cols_map[p]
        return None

    part_no_col = find_col([k for k in cols_map if 'PART' in k and ('NO' in k or 'NUM' in k)])
    desc_col = find_col([k for k in cols_map if 'DESC' in k])
    bus_model_col = find_col([k for k in cols_map if 'BUS' in k and 'MODEL' in k])
    station_no_col = find_col([k for k in cols_map if 'STATION' in k and 'NAME' not in k]) 
    station_name_col = find_col(['STATION NAME', 'ST. NAME', 'STATION_NAME', 'ST_NAME'])
    container_col = find_col([k for k in cols_map if 'CONTAINER' in k])
    qty_bin_col = find_col([k for k in cols_map if 'QTY/BIN' in k or 'QTY_BIN' in k or ('QTY' in k and 'BIN' in k)])
    qty_veh_col = find_col([k for k in cols_map if 'QTY/VEH' in k or 'QTY_VEH' in k or ('QTY' in k and 'VEH' in k)])
    zone_col = find_col(['ZONE', 'ABB ZONE', 'ABB_ZONE', 'AREA'])

    return {
        'Part No': part_no_col, 'Description': desc_col, 'Bus Model': bus_model_col,
        'Station No': station_no_col, 'Station Name': station_name_col,
        'Container': container_col, 'Qty/Bin': qty_bin_col,
        'Qty/Veh': qty_veh_col, 'Zone': zone_col
    }

def get_unique_containers(df, container_col):
    if not container_col or container_col not in df.columns: return []
    return sorted(df[container_col].dropna().astype(str).unique())

def automate_location_assignment(df, base_rack_id, rack_configs, status_text=None):
    required_cols = find_required_columns(df)
    
    if not all([required_cols['Part No'], required_cols['Container'], required_cols['Station No']]):
        st.error("❌ 'Part Number', 'Container Type', or 'Station No' column not found.")
        return None

    df_processed = df.copy()
    rename_dict = {v: k for k, v in required_cols.items() if v}
    df_processed.rename(columns=rename_dict, inplace=True)
    df_processed.sort_values(by=['Station No', 'Container'], inplace=True)

    final_parts_list = []
    
    for station_no, station_group in df_processed.groupby('Station No', sort=False):
        if status_text: status_text.text(f"Processing station: {station_no}...")

        rack_idx, level_idx = 0, 0
        sorted_racks = sorted(rack_configs.items())

        # Group by container within the station to fill racks logically
        for container_type, parts_group in station_group.groupby('Container', sort=True):
            items_to_place = parts_group.to_dict('records')
            
            while items_to_place:
                slot_found = False
                search_rack_idx, search_level_idx = rack_idx, level_idx
                
                # Search for a rack that has capacity for this container
                while search_rack_idx < len(sorted_racks):
                    rack_name, config = sorted_racks[search_rack_idx]
                    levels = config.get('levels', [])
                    capacity_per_level = config.get('rack_bin_counts', {}).get(container_type, 0)

                    if capacity_per_level > 0 and search_level_idx < len(levels):
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

                # Fill this level up to its specific capacity
                parts_for_level = items_to_place[:level_capacity]
                items_to_place = items_to_place[level_capacity:]
                
                # Determine how many "EMPTY" slots to add if the level isn't full
                num_empty_slots = level_capacity - len(parts_for_level)
                level_items = parts_for_level + ([{'Part No': 'EMPTY'}] * num_empty_slots)
                
                item_template = {col: '' for col in df_processed.columns}

                for cell_idx, item in enumerate(level_items, 1):
                    rack_num_val = ''.join(filter(str.isdigit, rack_name))
                    # Handle rack numbering (e.g., Rack 01 -> R1=0, R2=1)
                    rack_num_1st = rack_num_val[0] if len(rack_num_val) > 1 else '0'
                    rack_num_2nd = rack_num_val[1] if len(rack_num_val) > 1 else (rack_num_val[0] if rack_num_val else '1')
                    
                    location_info = {
                        'Rack': base_rack_id, 'Rack No 1st': rack_num_1st, 'Rack No 2nd': rack_num_2nd,
                        'Level': levels[level_idx], 'Cell': str(cell_idx), 'Station No': station_no,
                        'Rack Key': f"{rack_num_1st}{rack_num_2nd}"
                    }

                    if item['Part No'] == 'EMPTY':
                        full_item = item_template.copy()
                        full_item.update({'Part No': 'EMPTY', 'Container': container_type})
                    else:
                        full_item = item

                    full_item.update(location_info)
                    final_parts_list.append(full_item)

                # Move to next level
                level_idx += 1
                if level_idx >= len(levels):
                    level_idx = 0
                    rack_idx += 1
    
    if not final_parts_list: return pd.DataFrame()
    return pd.DataFrame(final_parts_list)

def create_location_key(row):
    return '_'.join([str(row.get(c, '')) for c in ['Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']])

def extract_location_values(row):
    # Returns list: [Bus Model, Station No, Rack, R1, R2, Level, Cell]
    return [str(row.get(c, '')) for c in ['Bus Model', 'Station No', 'Rack', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell']]


# --- PDF Generation (Rack Labels) ---
def generate_rack_labels_v1(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count = 0
    label_summary = {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        if status_text: status_text.text(f"Processing Rack Label {i+1}/{total_locations}")
        
        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No', '')).upper() == 'EMPTY': continue

        rack_key = f"ST-{part1.get('Station No', 'NA')} / Rack {part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())
        
        part2 = group.iloc[1] if len(group) > 1 else part1
        
        part_table1 = Table([['Part No', format_part_no_v1(str(part1.get('Part No','')))], ['Description', format_description_v1(str(part1.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        part_table2 = Table([['Part No', format_part_no_v1(str(part2.get('Part No','')))], ['Description', format_description_v1(str(part2.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.3*cm, 0.8*cm])
        
        # --- Autosizing Logic for Location Values ---
        location_values = extract_location_values(part1)
        formatted_loc_values = []
        for idx, val in enumerate(location_values):
            col_type = 'Bus Model' if idx == 0 else 'Station No' if idx == 1 else 'Default'
            style = get_dynamic_location_style(str(val), col_type)
            formatted_loc_values.append(Paragraph(str(val), style))

        location_data = [[Paragraph('Line Location', location_header_style)] + formatted_loc_values]
        
        col_props = [1.8, 2.7, 1.3, 1.3, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_props) for w in col_props]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.8*cm)
        
        part_style = TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('FONTNAME', (0, 0), (0, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (0, -1), 16)])
        part_table1.setStyle(part_style)
        part_table2.setStyle(part_style)
        
        loc_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        loc_style_cmds = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
        for j, color in enumerate(loc_colors):
            loc_style_cmds.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(loc_style_cmds))
        
        elements.extend([part_table1, Spacer(1, 0.3 * cm), part_table2, Spacer(1, 0.3 * cm), location_table, Spacer(1, 1.2 * cm)])
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary

def generate_rack_labels_v2(df, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*cm, bottomMargin=1*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    
    df['location_key'] = df.apply(create_location_key, axis=1)
    df.sort_values(by=['Station No', 'Rack No 1st', 'Rack No 2nd', 'Level', 'Cell'], inplace=True)
    df_grouped = df.groupby('location_key')
    total_locations = len(df_grouped)
    label_count = 0
    label_summary = {}

    for i, (location_key, group) in enumerate(df_grouped):
        if progress_bar: progress_bar.progress(int((i / total_locations) * 100))
        if status_text: status_text.text(f"Processing Rack Label {i+1}/{total_locations}")

        part1 = group.iloc[0].to_dict()
        if str(part1.get('Part No', '')).upper() == 'EMPTY': continue
        
        rack_key = f"ST-{part1.get('Station No', 'NA')} / Rack {part1.get('Rack No 1st', '0')}{part1.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1
            
        if label_count > 0 and label_count % 4 == 0: elements.append(PageBreak())

        part_table = Table([['Part No', format_part_no_v2(str(part1.get('Part No','')))], ['Description', format_description(str(part1.get('Description','')))]], colWidths=[4*cm, 11*cm], rowHeights=[1.9*cm, 2.1*cm])
        
        location_values = extract_location_values(part1)
        formatted_loc_values = []
        for idx, val in enumerate(location_values):
            col_type = 'Bus Model' if idx == 0 else 'Station No' if idx == 1 else 'Default'
            style = get_dynamic_location_style(str(val), col_type)
            formatted_loc_values.append(Paragraph(str(val), style))

        location_data = [[Paragraph('Line Location', location_header_style)] + formatted_loc_values]

        col_widths = [1.7, 2.9, 1.3, 1.2, 1.3, 1.3, 1.3]
        location_widths = [4 * cm] + [w * (11 * cm) / sum(col_widths) for w in col_widths]
        location_table = Table(location_data, colWidths=location_widths, rowHeights=0.9*cm)
        
        part_table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 1, colors.black), ('ALIGN', (0, 0), (0, -1), 'CENTER'), ('ALIGN', (1, 1), (1, -1), 'LEFT'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('FONTNAME', (0, 0), (0, -1), 'Helvetica'), ('FONTSIZE', (0, 0), (0, -1), 16)]))
        
        loc_colors = [colors.HexColor('#E9967A'), colors.HexColor('#ADD8E6'), colors.HexColor('#90EE90'), colors.HexColor('#FFD700'), colors.HexColor('#ADD8E6'), colors.HexColor('#E9967A'), colors.HexColor('#90EE90')]
        loc_style_cmds = [('GRID', (0, 0), (-1, -1), 1, colors.black), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
        for j, color in enumerate(loc_colors):
            loc_style_cmds.append(('BACKGROUND', (j+1, 0), (j+1, 0), color))
        location_table.setStyle(TableStyle(loc_style_cmds))
        
        elements.extend([part_table, Spacer(1, 0.3 * cm), location_table, Spacer(1, 1.5 * cm)])
        label_count += 1
        
    if elements: doc.build(elements)
    buffer.seek(0)
    return buffer, label_summary


# --- PDF Generation (Bin Labels Helpers) ---
def generate_qr_code_image(data_string):
    if not QR_AVAILABLE: return None
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4)
    qr.add_data(data_string)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    img_buffer = BytesIO()
    qr_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    return RLImage(img_buffer, width=2.5*cm, height=2.5*cm)

def extract_store_location_data_from_excel(row_data):
    col_lookup = {str(k).strip().upper(): k for k in row_data.keys()}
    def get_clean_value(possible_names, default=''):
        for name in possible_names:
            clean_name = name.strip().upper()
            if clean_name in col_lookup:
                val = row_data.get(col_lookup[clean_name])
                if pd.notna(val) and str(val).strip().lower() not in ['nan', 'none', 'null', '']:
                    return str(val).strip()
        return default

    return [
        get_clean_value(['ST. NAME (Short)', 'ST.NAME (Short)', 'ST NAME (Short)', 'STATION NAME SHORT']),
        get_clean_value(['Store Location', 'STORELOCATION']),
        get_clean_value(['ABB ZONE', 'ABB_ZONE']),
        get_clean_value(['ABB LOCATION', 'ABB_LOCATION']),
        get_clean_value(['ABB FLOOR', 'ABB_FLOOR']),
        get_clean_value(['ABB RACK NO', 'ABB_RACK_NO']),
        get_clean_value(['ABB LEVEL IN RACK', 'ABB_LEVEL_IN_RACK'])
    ]

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
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(1.8)
        canvas.rect((STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2 + doc.leftMargin, STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm, CONTENT_BOX_WIDTH - 0.2*cm, CONTENT_BOX_HEIGHT)
        canvas.restoreState()

    for i, row in enumerate(df_filtered.to_dict('records')):
        if progress_bar: progress_bar.progress(int(((i+1) / total_labels) * 100))
        if status_text: status_text.text(f"Processing Bin Label {i+1}/{total_labels}")
        
        rack_key = f"ST-{row.get('Station No', 'NA')} / Rack {row.get('Rack No 1st', '0')}{row.get('Rack No 2nd', '0')}"
        label_summary[rack_key] = label_summary.get(rack_key, 0) + 1

        qr_data = f"Part: {row['Part No']}\nDesc: {row['Description']}\nLoc: {row['Station No']}-{row['Rack']}{row['Rack Key']}-{row['Level']}{row['Cell']}"
        qr_image = generate_qr_code_image(qr_data)
        
        content_width = CONTENT_BOX_WIDTH - 0.2*cm
        main_table = Table([
            ["Part No", Paragraph(str(row['Part No']), bin_bold_style)],
            ["Description", Paragraph(str(row['Description'])[:47], bin_desc_style)],
            ["Qty/Bin", Paragraph(str(row['Qty/Bin']), bin_qty_style)]
        ], colWidths=[content_width/3, content_width*2/3], rowHeights=[0.9*cm, 1.0*cm, 0.5*cm])
        main_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black),('ALIGN', (0,0),(-1,-1), 'CENTER'), ('VALIGN', (0,0),(-1,-1), 'MIDDLE')]))

        # Location Tables
        store_vals = extract_store_location_data_from_excel(row)
        line_vals = extract_location_values(row)
        
        col_props = [1.8, 2.4, 0.7, 0.7, 0.7, 0.7, 0.9]
        inner_widths = [w * (content_width * 2/3) / sum(col_props) for w in col_props]

        # Inner tables for formatting
        s_inner = Table([store_vals], colWidths=inner_widths, rowHeights=[0.5*cm])
        s_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))
        
        l_inner = Table([line_vals], colWidths=inner_widths, rowHeights=[0.5*cm])
        l_inner.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('FONTSIZE', (0,0),(-1,-1), 9), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        s_table = Table([[Paragraph("Store Loc", bin_desc_style), s_inner]], colWidths=[content_width/3, content_width*2/3])
        l_table = Table([[Paragraph("Line Loc", bin_desc_style), l_inner]], colWidths=[content_width/3, content_width*2/3])
        
        # MTM Models
        mtm_data = [mtm_models, [str(row['Qty/Veh']) if str(row['Bus Model']).upper() == m.upper() else "" for m in mtm_models]]
        mtm_table = Table(mtm_data, colWidths=[3.6*cm/max(1, len(mtm_models))]*len(mtm_models), rowHeights=[0.75*cm]*2)
        mtm_table.setStyle(TableStyle([('GRID', (0,0),(-1,-1), 1.2, colors.black), ('ALIGN', (0,0),(-1,-1), 'CENTER')]))

        bottom_row = Table([[mtm_table, "", qr_image or "", ""]], colWidths=[3.6*cm, 1.0*cm, 2.5*cm, content_width-7.1*cm], rowHeights=[2.5*cm])
        
        all_elements.extend([main_table, s_table, l_table, Spacer(1, 0.2*cm), bottom_row, PageBreak()])

    if all_elements: doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
    buffer.seek(0)
    return buffer, label_summary


# --- PDF Generation (Rack List) ---
def generate_rack_list_pdf(df, base_rack_id, top_logo_file, top_logo_w, top_logo_h, fixed_logo_path, progress_bar=None, status_text=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), topMargin=0.5*cm, bottomMargin=0.5*cm, leftMargin=1*cm, rightMargin=1*cm)
    elements = []
    
    df = df[df['Part No'].str.upper() != 'EMPTY'].copy()
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
        top_logo_img = RLImage(io.BytesIO(top_logo_file.getvalue()), width=top_logo_w*cm, height=top_logo_h*cm) if top_logo_file else ""
        
        header_table = Table([[Paragraph("Document Ref No.:", rl_header_style), "", top_logo_img]], colWidths=[5*cm, 17.5*cm, 5*cm])
        elements.append(header_table)
        elements.append(Spacer(1, 0.1*cm))
        
        master_data = [
            [Paragraph("STATION NAME", rl_header_style), Paragraph(str(first_row.get('Station Name', '')), master_value_style_left), Paragraph("STATION NO", rl_header_style), Paragraph(str(station_no), master_value_style_center)],
            [Paragraph("MODEL", rl_header_style), Paragraph(str(first_row.get('Bus Model', '')), master_value_style_left), Paragraph("RACK NO", rl_header_style), Paragraph(f"Rack - {rack_key}", master_value_style_center)]
        ]
        master_table = Table(master_data, colWidths=[4*cm, 9.5*cm, 4*cm, 10*cm], rowHeights=[0.8*cm, 0.8*cm])
        master_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,1), colors.HexColor("#8EAADB"))]))
        elements.append(master_table)
        
        header_row = ["S.NO", "PART NO", "PART DESCRIPTION", "CONTAINER", "QTY/BIN", "LOCATION"]
        col_widths = [1.5*cm, 4.5*cm, 9.5*cm, 3.5*cm, 2.5*cm, 6.0*cm]
        data_rows = [header_row]
        
        for idx, row in enumerate(group.to_dict('records'), 1):
            loc_str = f"{row['Bus Model']}-{row['Station No']}-{base_rack_id}{rack_key}-{row['Level']}{row['Cell']}"
            data_rows.append([str(idx), str(row['Part No']), Paragraph(str(row['Description']), rl_cell_left_style), str(row['Container']), str(row['Qty/Bin']), loc_str])

        data_table = Table(data_rows, colWidths=col_widths)
        data_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F4B084"))]))
        elements.append(data_table)
        
        # Footer
        f_logo = RLImage(fixed_logo_path, width=4.3*cm, height=1.5*cm) if os.path.exists(fixed_logo_path) else ""
        footer_table = Table([[Paragraph(f"Date: {datetime.date.today()}", rl_cell_left_style), f_logo]], colWidths=[20*cm, 7.7*cm])
        elements.append(footer_table)
        elements.append(PageBreak())
        
    doc.build(elements)
    buffer.seek(0)
    return buffer, total_groups


# --- Main Application UI ---
def main():
    st.title("🏷️ AgiloSmartTag Studio")
    st.markdown("<p style='font-style:italic;'>Designed and Developed by Agilomatrix</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.title("📄 Label Options")
    output_type = st.sidebar.selectbox("Choose Output Type:", ["Bin Labels", "Rack Labels", "Rack List"])

    rack_label_format = "Single Part"
    if output_type == "Rack Labels":
        rack_label_format = st.sidebar.selectbox("Choose Rack Label Format:", ["Single Part", "Multiple Parts"])

    top_logo_file = None
    top_logo_w, top_logo_h = 3.0, 1.0
    if output_type == "Rack List":
        st.sidebar.markdown("**Rack List Configuration**")
        top_logo_file = st.sidebar.file_uploader("Upload Top Logo", type=['png', 'jpg', 'jpeg'])
        if top_logo_file:
            c1, c2 = st.sidebar.columns(2)
            top_logo_w = c1.slider("Logo Width (cm)", 1.0, 8.0, 4.0)
            top_logo_h = c2.slider("Logo Height (cm)", 0.5, 4.0, 1.5)

    model1, model2, model3 = "", "", ""
    if output_type == "Bin Labels":
        st.sidebar.markdown("**Enter up to 3 Vehicle Models**")
        model1 = st.sidebar.text_input("Vehicle Model 1", value="7M")
        model2 = st.sidebar.text_input("Vehicle Model 2", value="9M")
        model3 = st.sidebar.text_input("Vehicle Model 3", value="12M")

    base_rack_id = st.sidebar.text_input("Enter Storage Line Side Infrastructure", "R")
    
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['xlsx', 'xls', 'csv'])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, dtype=str).fillna('')
            st.success(f"✅ File loaded! Found {len(df)} rows.")
            
            required_cols_check = find_required_columns(df)
            
            if required_cols_check['Container']:
                unique_containers = get_unique_containers(df, required_cols_check['Container'])
                
                with st.expander("⚙️ Step 1: Configure Dimensions and Rack Setup (Applied to Each Station)", expanded=True):
                    st.subheader("1. Container Dimensions")
                    bin_dims = {c: st.text_input(f"Dimensions for {c}", key=f"bindim_{c}") for c in unique_containers}
                    
                    st.subheader("2. Rack Dimensions & Bin/Level Capacity")
                    num_racks = st.number_input("Number of Racks (per station)", min_value=1, value=1, step=1)
                    rack_configs = {}
                    for i in range(num_racks):
                        rack_name = f"Rack {i+1:02d}"
                        col1, col2 = st.columns(2)
                        with col1:
                            r_dim = st.text_input(f"Dimensions for {rack_name}", key=f"rdim_{i}")
                            levels = st.multiselect(f"Levels for {rack_name}", ['A','B','C','D','E','F'], default=['A','B','C'], key=f"lev_{i}")
                        with col2:
                            st.markdown(f"**Capacity of Bins Per Level for {rack_name}**")
                            rack_bin_counts = {c: st.number_input(f"Capacity of '{c}'", 0, 20, 1, key=f"cap_{i}_{c}") for c in unique_containers}
                        rack_configs[rack_name] = {'levels': levels, 'rack_bin_counts': rack_bin_counts}

                if st.button("🚀 Generate PDF", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    df_processed = automate_location_assignment(df, base_rack_id, rack_configs, status_text)
                    
                    if df_processed is not None and not df_processed.empty:
                        if output_type == "Rack Labels":
                            gen_func = generate_rack_labels_v2 if rack_label_format == "Single Part" else generate_rack_labels_v1
                            pdf_buffer, label_summary = gen_func(df_processed, progress_bar, status_text)
                            count = sum(label_summary.values())
                        elif output_type == "Bin Labels":
                            mtm_models = [m for m in [model1, model2, model3] if m]
                            pdf_buffer, label_summary = generate_bin_labels(df_processed, mtm_models, progress_bar, status_text)
                            count = sum(label_summary.values())
                        elif output_type == "Rack List":
                            pdf_buffer, count = generate_rack_list_pdf(df_processed, base_rack_id, top_logo_file, top_logo_w, top_logo_h, "Image.png", progress_bar, status_text)

                        st.download_button("📥 Download PDF", pdf_buffer.getvalue(), f"Agilo_{output_type}.pdf", "application/pdf")
            else:
                st.error("❌ Container column not found.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
