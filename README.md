# CVCL

The manuscript titled "Deep Multiview Clustering by Contrasting Cluster Assignments" is accepted by ICCV 2023. Please feel free to contact me if you have any prbolem. 

[https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Deep_Multiview_Clustering_by_Contrasting_Cluster_Assignments_ICCV_2023_paper.html)

You may reproduce the experimental results presented in the paper if setting the following parameter "load_model" to True. This means that the program will load the trained model provided by the authors for clustering tasks.

parser.add_argument('--load_model', default=False, help='Testing if True or training.')

You may consult the commented code in the file “main.py” for guidance on selecting the proper parameters if the trained model is unavailable for additional datasets.

Thank you.

--Prerequisites

Linux

--Required Python Packages

python>=3.9.7

pytorch>=1.7.1

numpy>=1.21.5

scikit-learn>=1.0.1

scipy>=1.7.3
