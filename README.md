SGUSCHI is a fork of SLUSCHI for simulating an oxidation environment using the small-cell methodology.

## To-Do

OxidationStep
+ In oxidationstep, we assume that the folder number is equal to rateanalysis, would be good to check if this is the case with a function.
+ Add runtime checks to ensure things are running smoothly

Testcases
+ Implement better test cases (3 cases, normal counting O2, adding an O2, Removing various gasses)

PrepareWorkPlace
+ PrepareWorkPlace to check POTCAR, POSCAR, CovalentRadii to see that they all match
+Prepareworkplace should run the first job so volsearch_cont works
+ PrepareWorkPlace to read job.in for job submission command
+ PrepareWorkPlace to create template OxidationMaster
+ PrepareWorkPlace should ensure INCAR nsw is set to 80
+ make Md steps 10000000!

OxidationPreProcessing
+ Add Supercell generation
+ Add SQS (Read Paper first)

Requirements.txt
+ Add it

SLUSCHI_mod
+ Change .sluschi.rc / make so that there is a sluschipath2, then you can install both sluschi and sguschi without conflict. Make sure all references to sluschipath are changed to sluschipath2 or sguschippath
+ make it such that volsearch_cont stops when OxidationStep raises an exception

Misc.
+ Finish re-doing compiling VASP apptainers


Steps to install / run:
1. Customise oxidation master
2. run make in sluschimod
3. chmod +x * in sluschimod
4. Add required files to a folder (list soon)
5. Run PrepareWorkPlace.py in folder (if Dry run (currently only implemented)
then run first job in each Dir_VolSearch folder)
6. Customise OxidationMaster
7. Submit OxidationMaster to cluster