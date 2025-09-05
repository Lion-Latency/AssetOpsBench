# Track 1 Submission Guidelines

This document provides the steps required to prepare, test, and submit your work to **CodaBench** for the **AssetOpsBench Competition**, more specific for **Track 1**.

---

## 0. Prerequisites for Track 1 Competition

Please ensure the competition code is ready on your local machine.  
Refer to the [README_CODS.md](https://github.com/IBM/AssetOpsBench/blob/Competition_CODS/benchmark/cods_track1/README_CODS.md) for details.

1. **Clone the Repository Locally**  
   - Use GitHub Desktop, CLI, or another tool to clone your fork of [IBM/AssetOpsBench](https://github.com/IBM/AssetOpsBench).  
   - Make sure you are on the **`Competition_CODS`** branch.  
   - Recommended local path:  
     ```
     <your_local_directory>/codabench/AssetOpsBench
     ```

2. **Verify Setup**  
   
   - Confirm that the branch is set to `Competition_CODS`.  
   - Your local files should match the competition branch.
   
3. **Code Editing Environment**  
   - You can use **Visual Studio Code** or any other IDE/text editor to view, modify, and test the code locally.



## 1. Understanding the Track 1 Competion

### Access CodaBench Competition Page

1. Go to the website of Codabench

   👉 [https://www.codabench.org/]  

2. Log in with your account credentials.  

3. Navigate to the **AssetOpsBench Competition page**:  
   👉 [https://www.codabench.org/competitions/10206]  

4. At the left side, click under the Started of the link for Track 1: Planning

   []: 

    



---


## 2. Local Testing (Optional but Recommended)
Before submitting, you can test your code locally to ensure it runs as expected.

1. Add required **environment variables** to your system (API keys, credentials, etc.).  
2. Run the workflow using the provided script. For example:  

   ```bash
   python run_track_1.py --utterance_ids <utterance_id1>,<utterance_id2>
   ```

3. Alternatively, edit **`entrypoint.sh`** to point to your script and run locally:  

   ```bash
   python /home/your_file_name.py --utterance_ids 108
   ```

This will help confirm that your code executes correctly before submission.

---

## 1. Prepare Your Submission
1. Package your solution as a **ZIP file**.  
   - Ensure the ZIP contains all required files (e.g., your Python scripts, configuration files, and any dependencies as specified in the task instructions).  
   - Double-check that your main script matches the expected entrypoint.  

---

## 4. Final Submission
1. On the competition page, go to **“My Submissions.”**  
2. Click **“Submit”** → **“Upload.”**  
3. Select your prepared **ZIP file.**  
4. Confirm the upload.  

---

## 5. After Submission
- Your job will automatically be placed in the competition pool; you do **not** need to create a new pool.  
- Once uploaded, the system will execute your submission.  
- Track the progress and results on the **My Submissions** page.  

---

✅ **Tip:** Keep a copy of your ZIP and logs for troubleshooting or future re-submissions.
