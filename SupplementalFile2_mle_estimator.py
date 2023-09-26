
### This software is distributed as-is without any waranty.
### Please see the CC0 1.0 Universal License distributed in this repository or visit https://creativecommons.org/publicdomain/zero/1.0/.

##### Code for calculating mle for AMHIRs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import sys
###############################################
###### Step 0 Read data
###############################################

df = pd.read_csv(sys.argv[1],sep="\t")
df['Recomb_bp'] = df['Recomb'] * 1e-8
df_95 = df[df['P_AMHIR'] > 0.95]  ### get high confidence AMHIRs
df_95 = df_95.reset_index(drop=True)

###############################################
###### Step 1 calculate non-truncated average
###############################################
truncation_point = int(sys.argv[2]) #50000  ## minimum segment length

def ln_observation_prob(length, rate_parameter, tp=truncation_point):
    raw_prob = stats.expon.pdf(length, scale=rate_parameter) # prob data given exponential distribution
    scaled_prob = raw_prob/(1-stats.expon.cdf(tp, scale=rate_parameter)) # adjust for truncation
    return(np.log(scaled_prob))

def ln_data_likelihood(lengths, rate_parameter, tp=truncation_point):
    lnlike = np.sum([ln_observation_prob(length, rate_parameter, tp) for length in lengths])
    return lnlike

#### Estimate the non-truncated distribution
xs = np.linspace(10000,150000, num=500)
ys = [ln_data_likelihood(df_95['Size'], rate, truncation_point) for rate in xs]

## maximum likelihood estimator of the rate parameter
ml_est = xs[np.argmax(ys)]
#print('Maximum likelihood: {:.2}'.format(ml_est), ml_est)
## CI (+-2 log likelihood units)
ys = ys-np.amax(ys)+2
xs = xs[ys>0]

LCI = xs[0]
UCI = xs[-1]


###############################################
##### step 2 calculate the amount of
#####  missing high confidence AMHIRs
###############################################
def calculate_missingAMHIRs(rate,num):
    X = truncation_point #50000
    t = truncation_point #50000
    r = rate
    density_calc = 1 - np.exp(-rate*X)
    Calc2 = ( 1 - np.exp( -r*t ) * ( 1 + r*t ) ) / r
    n = num
    p = density_calc
    m = (n*p) / (1-p)
    T = Calc2 * m
    return(density_calc,Calc2,m,T)

# m = np/(1-p) where n is the number of NHRs that were called and p is the proportion of true NHRs that are missing.
#That number m times the mean length will estimate the total missing length of introgressed material.
VALS = calculate_missingAMHIRs(1/float(ml_est), len(df_95))
P_missing = VALS[0]
AVG_missing = VALS[1]
Num_missing = VALS[2]
Size_missing = VALS[3]

###############################################
##### step 3 calculate AMHIR proportion
###############################################

def calculate_AMHIRSequence(df_All,df_HighConf,Missing_High):
    Total_High = sum(df_HighConf['Size'])
    Total_High_NonTruncated = Total_High + Missing_High

    df_All['SIZE_AMHIR'] = df_All['Size'] * df_All['P_AMHIR']
    Total_AMHIR_like = sum(df_All['SIZE_AMHIR'])

    P_HighConf = Total_High_NonTruncated/Total_High
    Total_AMHIR = P_HighConf * Total_AMHIR_like
    
    return(Total_High_NonTruncated,Total_AMHIR,Total_AMHIR_like,P_HighConf)

VALS = calculate_AMHIRSequence(df,df_95,Size_missing)
AMHIR_HighConf_Size = VALS[0]
AMHIR_Size = VALS[1]
AMHIR_Like = VALS[2]
AMHIR_HighConf_P = VALS[3]
Genome_Size = 337146633

Proportion_AMHIR = AMHIR_Size/Genome_Size

###############################################
##### step 4 calculate time of introgression
###############################################
def Introgression_Time(AVG, migration, recombination):
    r = recombination
    m = migration
    T = (1/((1-m) * r * AVG)) + 1
    return(T)

Altai = 122000 # age of altai neanderthal in years
Gen = 29       # human generation time
RecombinationRate = np.mean(df_95['Recomb_bp']) #1e-8 #float(sys.argv[2]) ### average recombination rate at AMHIRs

Time_MLE = Introgression_Time(AVG=ml_est, migration=Proportion_AMHIR, recombination=RecombinationRate)
Time_L = Introgression_Time(AVG=LCI, migration=Proportion_AMHIR, recombination=RecombinationRate)
Time_U = Introgression_Time(AVG=UCI, migration=Proportion_AMHIR, recombination=RecombinationRate)

Yrs = Time_MLE * Gen + Altai
Yrs_L = Time_L * Gen + Altai
Yrs_U = Time_U * Gen + Altai

###############################################
##### step 5 Output Values
###############################################

write_file = open(sys.argv[3],"w")

write_file.write("Proportion_NonTrunc_Trunc" + "\t" + "HighConfidence_AMHIR" + "\t" + "Missing_HighConfidenceAMHIR" + "\t" + "AMHIRLikeSize" +  
                 "\t" + "Total_AMHIR" +  "\t" + "Genome_Size" + "\t" + "P_Introgression" + "\t" + "AVG_Recomb" + 
                 "\t" + "MLE" + "\t" + "MLE_LCI" + "\t" + "MLE_UCI" + 
                 "\t" + "Time_gen" + "\t" + "Time_LCI_gen" + "\t" + "Time_UCI_gen" +
                 "\t" + "Time_yrs" +  "\t" + "Time_LCI_yrs" + "\t" + "Time_UCI_yrs" + "\n")

write_file.write(str(AMHIR_HighConf_P) + "\t" + str(sum(df_95['Size']))  + "\t" + str(int(round(Size_missing,0))) + "\t" + str(AMHIR_Like) +"\t" + 
                 str(AMHIR_Size) + "\t" + str(Genome_Size) + "\t" +  str(round(Proportion_AMHIR,4)) + "\t" +  str(round(RecombinationRate*1e8,4)) + "\t" + 
                 str(int(round(ml_est,0))) + "\t" + str(int(round(LCI,0))) + "\t" + str(int(round(UCI,0))) + "\t" +
                 str(int(round(Time_MLE,0))) + "\t" + str(int(round(Time_L,0))) + "\t" + str(int(round(Time_U,0))) + "\t" + 
                 str(int(round(Yrs,0))) + "\t" + str(int(round(Yrs_U,0))) + "\t" + str(int(round(Yrs_L,0))) + "\n")
write_file.close()


##### Code for calculating mle for NIRs
##### Same as for AMHIRs, except put posterior probabilities in terms of being an NIR: df['P_NIR'] = 1 - df['P_AMHIR']
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import sys
###############################################
###### Step 0 Read data
###############################################

df = pd.read_csv(sys.argv[1],sep="\t")
df['P_NIR'] = 1 - df['P_AMHIR']
print(df)
df['Recomb_bp'] = df['Recomb'] * 1e-8
df_95 = df[df['P_NIR'] > 0.50]  ### get high confidence AMHIRs
df_95 = df_95.reset_index(drop=True)

###############################################
###### Step 1 calculate non-truncated average
###############################################
truncation_point = int(sys.argv[2]) #50000  ## minimum segment length

def ln_observation_prob(length, rate_parameter, tp=truncation_point):
    raw_prob = stats.expon.pdf(length, scale=rate_parameter) # prob data given exponential distribution
    scaled_prob = raw_prob/(1-stats.expon.cdf(tp, scale=rate_parameter)) # adjust for truncation
    return(np.log(scaled_prob))

def ln_data_likelihood(lengths, rate_parameter, tp=truncation_point):
    lnlike = np.sum([ln_observation_prob(length, rate_parameter, tp) for length in lengths])
    return lnlike

#### Estimate the non-truncated distribution
xs = np.linspace(10000,150000, num=500)
ys = [ln_data_likelihood(df_95['Size'], rate, truncation_point) for rate in xs]

## maximum likelihood estimator of the rate parameter
ml_est = xs[np.argmax(ys)]
#print('Maximum likelihood: {:.2}'.format(ml_est), ml_est)
## CI (+-2 log likelihood units)
ys = ys-np.amax(ys)+2
xs = xs[ys>0]

LCI = xs[0]
UCI = xs[-1]


###############################################
##### step 2 calculate the amount of
#####  missing high confidence AMHIRs
###############################################
def calculate_missingAMHIRs(rate,num):
    X = truncation_point #50000
    t = truncation_point #50000
    r = rate
    density_calc = 1 - np.exp(-rate*X)
    Calc2 = ( 1 - np.exp( -r*t ) * ( 1 + r*t ) ) / r
    n = num
    p = density_calc
    m = (n*p) / (1-p)
    T = Calc2 * m
    return(density_calc,Calc2,m,T)

# m = np/(1-p) where n is the number of NHRs that were called and p is the proportion of true NHRs that are missing.
#That number m times the mean length will estimate the total missing length of introgressed material.
VALS = calculate_missingAMHIRs(1/float(ml_est), len(df_95))
P_missing = VALS[0]
AVG_missing = VALS[1]
Num_missing = VALS[2]
Size_missing = VALS[3]

###############################################
##### step 3 calculate AMHIR proportion
###############################################

def calculate_AMHIRSequence(df_All,df_HighConf,Missing_High):
    Total_High = sum(df_HighConf['Size'])
    Total_High_NonTruncated = Total_High + Missing_High

    df_All['SIZE_AMHIR'] = df_All['Size'] * df_All['P_NIR']
    Total_AMHIR_like = sum(df_All['SIZE_AMHIR'])

    P_HighConf = Total_High_NonTruncated/Total_High
    Total_AMHIR = P_HighConf * Total_AMHIR_like
    
    return(Total_High_NonTruncated,Total_AMHIR,Total_AMHIR_like,P_HighConf)

VALS = calculate_AMHIRSequence(df,df_95,Size_missing)
AMHIR_HighConf_Size = VALS[0]
AMHIR_Size = VALS[1]
AMHIR_Like = VALS[2]
AMHIR_HighConf_P = VALS[3]
Genome_Size = 337146633

Proportion_AMHIR = AMHIR_Size/Genome_Size

###############################################
##### step 4 calculate time of introgression
###############################################
def Introgression_Time(AVG, migration, recombination):
    r = recombination
    m = migration
    T = (1/((1-m) * r * AVG)) + 1
    return(T)

###############################################
##### step 5 Output Values
###############################################

write_file = open(sys.argv[3],"w")

write_file.write("Proportion_NonTrunc_Trunc" + "\t" + "HighConfidence_AMHIR" + "\t" + "Missing_HighConfidenceAMHIR" + "\t" + "AMHIRLikeSize" +  
                 "\t" + "Total_AMHIR" +  "\t" + "Genome_Size" + "\t" + "P_Introgression" + "\t"  +
                 "MLE" + "\t" + "MLE_LCI" + "\t" + "MLE_UCI" + "\n") 

write_file.write(str(AMHIR_HighConf_P) + "\t" + str(sum(df_95['Size']))  + "\t" + str(int(round(Size_missing,0))) + "\t" + str(AMHIR_Like) +"\t" + 
                 str(AMHIR_Size) + "\t" + str(Genome_Size) + "\t" +  str(round(Proportion_AMHIR,4)) + "\t"  + 
                 str(int(round(ml_est,0))) + "\t" + str(int(round(LCI,0))) + "\t" + str(int(round(UCI,0))) + "\n")
                 
write_file.close()

