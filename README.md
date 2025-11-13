SGUSCHI is a fork of SLUSCHI for simulating an oxidation environment using the small-cell methodology.

To-Do

OxidationStep
+ In oxidationstep, we assume that the folder number is equal to rateanalysis, would be good to check if this is the case with a function.

Testcases
+ Implement better test cases (3 cases, normal counting O2, adding an O2, Removing various gasses)

PrepareWorkPlace
+ PrepareWorkPlace to check POTCAR, POSCAR, CovalentRadii to see that they all match
+Prepareworkplace should run the first job so volsearch_cont works
+ PrepareWorkPlace to read job.in for job submission command
+ PrepareWorkPlace to create template master_jobsub
+PrepareWorkPlace should ensure INCAR nsw is set to 80
+ make Md steps 10000000!

OxidationPreProcessing
+ Add Supercell generation
+ Add SQS (Read Paper first)

Misc.
+ Finish re-doing compiling VASP apptainers


steps to install
update oxidation master
make in sluschimod
chmod +x * in sluschimod
