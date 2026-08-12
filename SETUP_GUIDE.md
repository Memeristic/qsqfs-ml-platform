# Setup Guide (written for a first-timer)

Three separate jobs. Do them in this order:

1. **Run it on your own computer** — you will use this the most.
2. **Put the code on GitHub** — a backup, and required for step 3.
3. **Publish the web app** — a link you can share.

You do not need step 2 or 3 to use the platform. Step 1 alone is enough.

Anything in a grey box is a command. Type it (or paste it) and press Enter.

---

# PART 1 — Run it on your own computer

## 1.1 Install Python

Go to <https://www.python.org/downloads/> and download Python 3.11 or newer.

**Windows users: on the first install screen, tick the box that says
"Add Python to PATH" before clicking Install.** It is easy to miss and nothing
below works without it.

Mac users: nothing special, just install it.

Check it worked. Open your terminal:

- **Windows** — press Start, type `cmd`, open **Command Prompt**
- **Mac** — press Cmd+Space, type `terminal`, open **Terminal**

Then type:

```
python --version
```

You should see something like `Python 3.11.5`. If Windows says "not
recognised", try `py --version` instead; if that works, use `py` everywhere
below in place of `python`.

## 1.2 Unzip the project somewhere you can find

Unzip `qsqfs-ml-platform.zip`. Put the folder somewhere simple, for example:

- Windows: `C:\Users\YourName\Documents\qsqfs-ml-platform`
- Mac: `/Users/YourName/Documents/qsqfs-ml-platform`

**Avoid folders with spaces or accents in the name** — it causes confusing
errors later.

## 1.3 Point the terminal at that folder

In your terminal, type `cd ` (the letters c and d, then a space), then drag the
project folder from your file explorer onto the terminal window and press Enter.
That fills in the path for you.

Check you are in the right place:

```
dir
```

(Mac: `ls`)

You should see `run_pipeline.py`, `app.py`, `README.md` and folders like `src`.
If you do not, you are in the wrong folder.

## 1.4 Create a virtual environment

This keeps the project's packages separate from the rest of your computer, so
nothing else breaks.

```
python -m venv .venv
```

Now switch it on:

**Windows:**
```
.venv\Scripts\activate
```

**Mac:**
```
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. That is how you know it is on.

> **You must do this every time you open a new terminal to work on this
> project.** If a command fails with "module not found", the usual cause is
> that you forgot this step. Just run it again.

## 1.5 Install the packages

```
pip install -r requirements.txt
```

This downloads a few hundred megabytes (PyTorch is large) and takes 2–10
minutes. It is normal for it to look stuck for a while.

## 1.6 Check everything works

```
pytest -q
```

You want to see **`63 passed`**. If you do, the installation is correct.

## 1.7 Do a first run with no data of your own

The project can generate its own practice data:

```
python scripts/make_fixture.py --out ./data/d1namo_fixture
python run_pipeline.py d1namo --data_root ./data/d1namo_fixture --n_iterations 10
```

You will see a results table. **Every model will score about the same as
`dummy_mean`.** That is correct and expected — the practice data is random
noise, so there is nothing to learn. It proves the tool does not invent
results.

Your results are saved in `run_results/`, in a folder named by date and time.
Open `summary.txt` inside it, and the `figures` folder for the charts.

## 1.8 Run it on your own spreadsheet

Put your CSV in the `data` folder, then:

```
python run_pipeline.py tabular --data_path ./data/my_file.csv --target my_column
```

Replace `my_file.csv` with your filename and `my_column` with the name of the
column you want to predict.

If the same patient or person appears in more than one row, add:

```
--group_col patient_id --cv_type group
```

(replacing `patient_id` with your ID column). This stops the same person
appearing in both the training and testing halves, which would make the results
look better than they really are.

## 1.9 Open the web app on your own computer

```
streamlit run app.py
```

Your browser opens automatically. Upload a spreadsheet and click **Run
pipeline**.

To stop it, click the terminal window and press **Ctrl+C**.

---

# PART 2 — Put the code on GitHub

## 2.1 Make a GitHub account

Sign up free at <https://github.com>.

## 2.2 Install GitHub Desktop

Download from <https://desktop.github.com>.

I am recommending the app rather than typed `git` commands. It is far easier
when you are starting out, and this project has over 100 files — more than
GitHub's website lets you drag-and-drop in one go.

Open it and sign in with your GitHub account.

## 2.3 Create the repository

In GitHub Desktop:

1. **File → Add Local Repository**
2. Choose your `qsqfs-ml-platform` folder
3. It will say it is not a git repository — click **create a repository**
4. Name: `qsqfs-ml-platform`
5. **Leave "Git ignore" set to None** — the project already has a `.gitignore`
   file that does the right thing
6. Click **Create Repository**

## 2.4 Check what is about to be uploaded

Look at the file list on the left. You should see code files.

**You should NOT see:**

- anything inside `data/`
- anything inside `run_results/`
- a `.venv` folder

If you do see those, stop and tell someone before continuing. The `.gitignore`
file is meant to exclude them. Uploading data can breach patient
confidentiality, and GitHub rejects files over 100 MB anyway.

## 2.5 Upload

1. Bottom left, in the **Summary** box, type: `Initial commit`
2. Click **Commit to main**
3. Top of the window, click **Publish repository**
4. **Untick "Keep this code private"** if you want the free Streamlit hosting
   (Community Cloud needs a public repo)
5. Click **Publish repository**

Your code is now at `https://github.com/YOUR-USERNAME/qsqfs-ml-platform`.

## 2.6 Later, when you change something

1. Open GitHub Desktop
2. Type a short note in the Summary box, e.g. `Fixed the target column`
3. Click **Commit to main**
4. Click **Push origin**

---

# PART 3 — Publish the web app

## 3.1 Nothing to swap

There is one `requirements.txt` and it works both locally and on Streamlit
Community Cloud. Do not rename it.

PyTorch is deliberately not in it. It is about 200 MB and Community Cloud
allows roughly 1 GB of memory, so including it makes hosted apps fail
unpredictably. Without it the hosted app still runs everything except the
Transformer — feature selection, all five baselines, tuning, every figure and
every statistical table — and says on screen that the Transformer was skipped.

On your own machine you add it once:

```
python -m pip install torch torchvision
```

That is the only difference between your local install and the hosted app.

## 3.2 Deploy

1. Go to <https://share.streamlit.io> and sign in **with GitHub**
2. Click **New app**
3. **Repository:** `YOUR-USERNAME/qsqfs-ml-platform`
4. **Branch:** `main`
5. **Main file path:** `app.py`
6. Click **Deploy**

First build takes 3–10 minutes. You will see a log scrolling; that is normal.

You get a URL like `https://qsqfs-ml-platform-yourname.streamlit.app`.

> If you already have an app at
> `qsqfs-diabetes-fs-obika.streamlit.app`, deploying this creates a **second,
> separate** app. To replace the old one instead, delete it from your Streamlit
> dashboard first, or just leave both running.

## 3.3 What the hosted app can and cannot do

**Can:** accept spreadsheet uploads, run the full analysis, show charts.

**Cannot:** run D1NAMO or PhysioCGM. Those are 10+ GB of raw recordings and
will never fit. There is no way around this on free hosting.

**But there is a way to still use them.** On your own computer, boil the
recordings down to a small table first:

```
python run_pipeline.py physiocgm --data_root ./data/PhysioCGM --export_features feats.csv --export_only
```

That produces a file of roughly 0.25 MB. Upload **that** to the hosted app like
any ordinary spreadsheet, and set the group column to `subject_id` and the
split strategy to `group`. The results are identical to running it locally.

**A word of caution:** if your data contains real patient information, do not
upload it to a public web app, even a small summary table. Use the app on your
own computer instead (`streamlit run app.py`), where nothing leaves your
machine.

---

# When something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `python is not recognised` | Python is not on your PATH | Reinstall Python, tick "Add Python to PATH". Or try `py` instead of `python` |
| `No module named pandas` | The virtual environment is off | Run the activate command from step 1.4 again |
| `No module named src` | You are in the wrong folder | `cd` into the project folder — you should see `run_pipeline.py` when you list files |
| `Target 'x' not in file` | Column name typo | Column names are case-sensitive. The error prints the real names — copy one exactly |
| `data_root does not exist` | Wrong path to the data | Check the folder exists and the spelling matches |
| Streamlit app "over resource limits" | Too much memory | You probably skipped step 3.1. Swap the requirements file |
| Every model matches the dummy | Usually not a bug | The data may genuinely hold no signal. See below |

## Reading your results honestly

Open `run_results/run_<date>/summary.txt` and check these four things, in
order:

1. **Does your model beat `dummy_majority` or `dummy_mean`?** If not, stop.
   Nothing else in the file matters. This is a real finding, not a failure —
   it means these features do not predict this outcome.
2. **Does the confidence interval overlap the best baseline?** If it does, you
   have not shown your model is better.
3. **Is anything listed as FLAGGED?** Decide for each whether it is a genuine
   measurement or something that gives away the answer.
4. **Does it say `group leakage: NONE`?** If not, rerun with
   `--cv_type group`.

A tool that tells you your model did not work is doing its job. Reporting a
number that came from a leak is far worse than reporting no result.
