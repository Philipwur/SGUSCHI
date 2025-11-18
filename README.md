SGUSCHI is a fork of SLUSCHI for simulating an oxidation environment using the small-cell methodology.

## To-Do

Test ReadRateAnalysis (are Gas species properly read into arrays?)
Update WriteXYZ to return fractional coordinates and change TimeFs to Time (readable by ovito)

Important, fix indexing checks on oxidationstep. len(RateAnalysis) should be one higher than current existing folder, because VolSearch_cont will create a new folder before oxidationstep is run.
+ VolSearch requires a folder where all old sims are in their folders, and new sim has yet to be moved into new folder
+ 

PrepareWorkPlace
+ PrepareWorkPlace to check POTCAR, POSCAR, CovalentRadii to see that they all match
+Prepareworkplace should run the first job so volsearch_cont works
+ PrepareWorkPlace to read job.in for job submission command
+ PrepareWorkPlace to create template OxidationMaster
+ PrepareWorkPlace should ensure INCAR nsw is set to 80
+ make Md steps 10000000!

Requirements.txt
+ Add it

SLUSCHI_mod
+ Change .sluschi.rc / make so that there is a sluschipath2/sguschipath, then you can install both sluschi and sguschi without conflict. Make sure all references to sluschipath are changed to sluschipath2 or sguschipath
+ make it such that volsearch_cont stops when OxidationStep raises an exception

utils/Rollback.py
+ create a function which rollsback simulation environment to last good folder (given by user or inferred ourselves (hard to do))
+ To do this, copy the POSCAR from the first "bad step", delete the bad step folder (and WAVCAR, CHG* in workdir), run jobsub, then when finished run volsearch_cont
+ POSCAR in bad folder is the end point of first good folder.

Misc.
+ Finish re-doing compiling VASP apptainers

--low prio TODO--

OxidationStep
+ Add more runtime checks to ensure things are running smoothly

OxidationPreProcessing
+ Add Supercell generation
+ Add SQS (Read Paper first)

Testcases
+ Implement better test cases (3 cases, normal counting O2, adding an O2, Removing various gasses)


Steps to install / run:
1. Customise oxidation master
2. run make in sluschimod
3. chmod +x * in sluschimod
4. Add required files to a folder (list soon)
5. Run PrepareWorkPlace.py in folder (if Dry run (currently only implemented)
then run first job in each Dir_VolSearch folder)
6. Customise OxidationMaster
7. Submit OxidationMaster to cluster