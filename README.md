# BlueMath Course at Oregon State University

**October 5–6, 2026**  
**Oregon State University — Corvallis, Oregon, USA**

Welcome to the repository for the **BlueMath Course at Oregon State University**.

This two-day hands-on course introduces **[BlueMath](https://github.com/GeoOcean/BlueMath)**, an open-source Python framework developed by the **GeoOcean Group at the University of Cantabria** to support efficient, reproducible, and modular workflows for coastal and ocean engineering applications.

The course combines short theoretical introductions with practical exercises using Jupyter notebooks. Participants will work directly with BlueMath tools and methodologies and learn how they can be adapted to different coastal applications.

---

## Repositories

- **Course repository:**  
  https://github.com/GeoOcean/BlueMath_Course_Corvallis

- **BlueMath repository:**  
  https://github.com/GeoOcean/BlueMath

The course repository contains all notebooks, exercises, datasets, and additional material required during the course.

The main BlueMath repository contains the source code, documentation, and general examples of the BlueMath framework.

---

## Course information

| | |
|---|---|
| **Dates** | October 5–6, 2026 |
| **Location** | Oregon State University, Corvallis, Oregon |
| **Format** | In-person, hands-on course |
| **Duration** | 2 days |
| **Language** | English |
| **Software** | Python, Jupyter, BlueMath |

---

## Course objectives

By the end of the course, participants will be able to:

- Understand the general philosophy and architecture of BlueMath.
- Work with the main BlueMath modules and workflows.
- Process and analyze environmental and coastal datasets using Python.
- Apply statistical and machine-learning techniques commonly used in coastal engineering.
- Develop reproducible workflows using Jupyter notebooks.
- Explore how BlueMath can be adapted to their own research and applications.

---

## Getting started

Clone the course repository:

```bash
git clone https://github.com/GeoOcean/BlueMath_Course_Corvallis.git
cd BlueMath_Course_Corvallis
```

The course material is based on the BlueMath framework available at:

```text
https://github.com/GeoOcean/BlueMath
```

Detailed installation instructions and the required Python environment will be provided in this repository.

---

## Repository structure

The repository contains the material required for the different practical sessions of the course.

A typical structure is:

```text
.
├── notebooks/          # Course notebooks and practical exercises
├── data/               # Data used during the course
├── examples/           # Additional examples
├── environment/        # Environment and installation files
├── figures/            # Figures and supporting material
└── README.md
```

The repository will be progressively updated with the material required for each session.

---

## Course program

### Day 1 — Statistical Modeling & Climate Emulation

*From historical climate data to synthetic climate*

| Time | Topic | Concepts | BlueMath Notebooks |
|---|---|---|---|
| 9:00–9:10 | Welcome | | |
| 9:10–9:30 | Introduction to BlueMath | What is BlueMath? | |
| 9:30–10:00 | Global Databases | How to extract databases from servers | `BlueMath/toolkit/data`<br>`Download_GEBCO_bathymetry_data.ipynb`<br>`SeaLevel_Data.ipynb` |
| 10:00–10:30 | Reducing Climate Complexity | PCA — How can I obtain the dominant climate variability from a dataset? | `BlueMath/toolkit/datamining` |
| 10:30–10:45 | Break | | |
| 10:45–12:00 | Weather typing / classification | KMA, weather types — How do I identify recurring climate states? | |
| 12:00–1:30 | Lunch Break | | |
| 1:30–2:15 | Statistical Downscaling | Regression (linear/WT) | `BlueMath/methods/statistical_downscaling` |
| 2:15–3:15 | Climate emulation | ALR / stochastic simulation of weather types | `ALR-AWT`, `ALR-DWT` |
| 3:15–3:30 | Break | | |
| 3:30–4:30 | Hands-on climate emulator | Build a long synthetic climate sequence + diagnostics | |

### Day 2 — Hybrid Statistical–Numerical Modeling

*From climate forcing to local impacts and risk*

| Time | Topic | Concepts | BlueMath Notebooks |
|---|---|---|---|
| 9:00–9:45 | Wave Downscaling — Why hybrid modeling? | Link emulator-hybrid / Statistical vs dynamical vs hybrid downscaling | |
| 9:45–10:30 | Running numerical models with BlueMath | Model wrappers, parallelization | |
| 10:30–10:45 | Break | | |
| 10:45–11:45 | Hands-on BinWaves | | |
| 11:45–1:15 | Lunch Break | | |
| 1:15–2:00 | Design of numerical experiments and metamodeling | Sampling / Selection / RBF / GPR | `BlueMath/toolkit/datamining`<br>`MDA_vs_MKeans.ipynb` |
| 2:00–3:00 | Hands-on HySwash | | |
| 3:00–3:30 | HySwash applications | CHySwash + Veggie | |
| 3:30–3:45 | Break | | |
| 3:45–4:15 | HyFlood | | |
| 4:15–5:00 | Exploring Climate Services | | |

The detailed schedule and associated notebooks will be available in this repository.

---

## BlueMath

**[BlueMath](https://github.com/GeoOcean/BlueMath)** is an open-source framework designed to provide reusable computational tools for coastal and ocean applications.

Its philosophy is based on the development of **independent and reusable building blocks** that can be combined to construct customized workflows for different scientific and engineering problems.

BlueMath includes methodologies related to:

- Environmental and climate data analysis
- Dimensionality reduction
- Clustering and classification
- Statistical modeling
- Weather Types
- Climate emulation
- Surrogate and hybrid modeling
- Coastal hazard assessment

Its modular architecture facilitates the development of transparent, reproducible, and transferable workflows and allows users to combine existing methodologies or incorporate new components depending on the application.

---

## Requirements

Basic experience with Python is recommended.

Familiarity with some of the following libraries will be useful, but is not required:

```text
numpy
pandas
xarray
matplotlib
scipy
scikit-learn
```

All required packages and installation instructions will be provided as part of the course material.

---

## Course material

The notebooks are designed to be executed during the practical sessions.

They include:

- Short theoretical introductions
- Methodological explanations
- Practical examples
- Code exercises
- Applications to coastal and environmental datasets

Participants are encouraged to modify the examples and explore the methodologies using their own datasets after the course.

---

## Instructors

The course is organized by the **GeoOcean Group at the University of Cantabria** in collaboration with **Oregon State University**.

- Fernando J. Méndez (Universidad de Cantabria)
- Laura Cagigal (Universidad de Cantabria)
- Pablo Alonso-Alguacil (Universidad de Cantabria)
- Jared Ortiz Angulo (Universidad de Cantabria)
- Peter Ruggiero (OSU)
- Alba Ricondo (OSU)

---

## Acknowledgements

We thank **Oregon State University** for hosting the course and supporting the dissemination and development of open-source tools for coastal and ocean science.

---

## License

The course material is intended for educational and research purposes.

For information about the license and conditions of use of BlueMath, please refer to the main **[BlueMath repository](https://github.com/GeoOcean/BlueMath)**.

---

## Contact

For questions related to the course, please open an issue in the **[course repository](https://github.com/GeoOcean/BlueMath_Course_Corvallis/issues)**.

For questions, issues, or contributions related to BlueMath itself, please visit the **[BlueMath repository](https://github.com/GeoOcean/BlueMath)**.

## Agenda — Adapted summary

- **Objective:** A two-day hands-on course introducing the `BlueMath` framework, its main modules, and reproducible workflows applied to coastal and ocean data.
- **Dates and location:** October 5–6, 2026 — Oregon State University, Corvallis.
- **Target audience:** Researchers, graduate students, and professionals with basic Python experience interested in environmental data analysis and coastal applications.

- **General structure:**
  - Day 1 — Statistical Modeling & Climate Emulation (from historical climate data to synthetic climate): global databases, PCA, weather typing (KMA), statistical downscaling, and climate emulation (ALR) with a hands-on session building a climate emulator.
  - Day 2 — Hybrid Statistical–Numerical Modeling (from climate forcing to local impacts and risk): wave downscaling, numerical model wrappers, design of numerical experiments and metamodeling, and hands-on sessions with BinWaves, HySwash/CHySwash+Veggie, and HyFlood, ending with Climate Services.

- **Material and requirements:** Notebooks, datasets, and exercises will be available in the repository. Basic knowledge of Python and libraries such as `numpy`, `pandas`, `xarray`, `matplotlib`, and `scikit-learn` is recommended.
- **Logistics:** The course is in-person and hands-on oriented; instructions for installing the Python environment and dependencies will be provided in the `environment` folder of the repository.
- **Expected outcomes:** By the end of the course, participants will be able to use `BlueMath` to build reproducible workflows, adapt methodologies to their own data, and prototype statistical models or emulators for coastal problems.

- **Contact:** Open an issue in the [course repository](https://github.com/GeoOcean/BlueMath_Course_Corvallis/issues) for specific questions or requests.
