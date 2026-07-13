### 1. Environment Setup (One Time Only)

```bash
# Virtual Environment banana (Sirf pehli baar)
python -m venv venv

# Virtual Environment activate karna (Har baar jab terminal kholo)
# Windows ke liye:
venv\Scripts\activate

```

### 2. Libraries Install (only one time)

```bash
pip install -r requirements.txt

```

### 3. Data Processing (Whenever we enter new data sheet)

```bash
python src/data_pipeline.py

```

### 4. Dashboard Run (When we want project to shown in browser)


```bash
streamlit run app.py

```

---

### Pro-Tips:

* **Har baar jab aap computer restart karke wapas kaam shuru karo:** Sab se pehle `cd` karke project folder mein jao, phir `venv` activate karo (Step 1 wali command), aur phir `streamlit run app.py` chala do.
* **Agar error aaye "ModuleNotFoundError":** Iska matlab hai aapne `venv` activate nahi kiya. Activate karo, error khatam ho jayega.
* **Agar code mein badlav karo:** Code save karne ke baad browser mein "Rerun" ya page refresh kar lena, aapki changes update ho jayengi.
