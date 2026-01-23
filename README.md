# 🏷️ AgiloSmartTag Studio

AgiloSmartTag Studio is a **Streamlit-based application** used to generate **Rack Labels**, **Bin Labels (with QR codes)**, and **Rack Lists** from Excel/CSV input data.  
It automates **station-wise rack allocation**, **cell assignment**, and **PDF generation** for shop-floor and warehouse labeling.

---

## 📌 Features

- Upload Excel / CSV data
- Automatic detection of required columns
- Station-wise rack and cell allocation
- Container-based capacity handling
- Sequential location ID generation
- PDF generation for:
  - Rack Labels
  - Bin Labels (with QR codes)
  - Rack Lists
- Configurable layouts via UI

---

## 📂 Input Data Requirements

The input file should contain (column names are auto-detected):

- Part Number
- Description
- Station Number
- Bus Model
- Container Type
- Optional: Quantity, Zone, Store Location fields

---

## 🔁 Overall Process Flow

```mermaid
flowchart TD
    A[Start Application] --> B[User Configures UI Options]
    B --> C[Upload Excel / CSV File]

    C --> D[Load File into Pandas DataFrame]
    D --> E[Detect Required Columns]

    E -->|Missing Station or Container| X[Show Error & Stop]
    E -->|Valid Columns| F[Global Rack Configuration]

    F --> G[Define Levels & Cells per Level]
    G --> H[Define Container Dimensions & Capacity]

    H --> I[Generate Station-wise Assignment]

    I --> I1[Group Data by Station]
    I1 --> I2[Sort Containers by Bin Area]
    I2 --> I3[Calculate Cells Required]
    I3 --> I4[Allocate Racks, Levels, Physical Cells]
    I4 --> I5[Assign Parts to Cells]
    I5 --> I6[Fill Remaining Cells as EMPTY]

    I6 --> J[Assign Sequential Location IDs]

    J --> K{Selected Output Type?}

    K -->|Rack Labels| L[Generate Rack Labels PDF]
    K -->|Bin Labels| M[Generate Bin Labels PDF with QR Code]
    K -->|Rack List| N[Generate Rack List PDF]

    L --> O[Show Summary & Download PDF]
    M --> O
    N --> O

    O --> P[End]

