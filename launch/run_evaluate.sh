#!/bin/bash

# model name
CPX1=dac_busi_056.pth

# method type
TP1=NoBRS

# dataset name
DT1=BUSI
DT2=BUSI1800
DT3=LITS
DT4=CHAOS_CT
DT5=MSD_Brain
DT6=BraTS_2018
DT7=LITS_Tumor
DT8=Btcv_tcia


# execute scripts
##############################################################################################
# BUSI
# python scripts/evaluate_model.py $TP1 --checkpoint=$CPX1 --datasets=$DT1
# python scripts/evaluate_model.py $TP1 --checkpoint=$CPX1 --datasets=$DT1 --vis-preds
python scripts/evaluate_model.py $TP1 --checkpoint=$CPX1 --datasets=$DT1 --es-analysis --vis-preds