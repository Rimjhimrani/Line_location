flowchart TD
    A[Start Streamlit App] --> B[UI Configuration]
    B --> C[Select Output Type]
    B --> D[Upload Excel / CSV]

    D --> E[Load Data into Pandas]
    E --> F[Find Required Columns]

    F -->|Missing Station or Container| X[Show Error & Stop]
    F -->|Valid| G[Global Rack Settings]

    G --> H[Container Rules\nDimensions & Capacity]
    H --> I[Generate Station-wise Assignment]

    I --> I1[Group by Station]
    I1 --> I2[Calculate Required Cells]
    I2 --> I3[Allocate Racks, Levels, Cells]
    I3 --> I4[Assign Parts]
    I4 --> I5[Fill EMPTY Cells]

    I5 --> J[Assign Sequential Location IDs]

    J --> K{Output Type?}

    K -->|Rack Labels| L[Generate Rack Label PDF]
    K -->|Bin Labels| M[Generate Bin Label PDF\n+ QR Codes]
    K -->|Rack List| N[Generate Rack List PDF]

    L --> O[Show Summary & Download]
    M --> O
    N --> O

    O --> P[End]
