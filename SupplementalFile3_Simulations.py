
### This software is distributed as-is without any warranty.
### Please see the CC0 1.0 Universal License distributed in this repository or visit https://creativecommons.org/publicdomain/zero/1.0/.

import tskit
import msprime
import numpy
import sys

# Population IDs
Den, Altai, AFR= 0, 1, 2
Gen = 29
def archaic_introgression(pop_config, dem_events,random_seed=None):
    ts = msprime.simulate(
        Ne=20000,
        recombination_rate=1e-8,
        mutation_rate=1e-8,
        length = 100000000, # 100 MB
        
        samples = [msprime.Sample(time=0, population=AFR)]*30 + [msprime.Sample(time=(122000/Gen), population=Altai)]*2 + [msprime.Sample(time=(76000/Gen), population=Den)]*2 ,
        
        population_configurations = pop_config,
        demographic_events = dem_events,                     
        
        record_migrations=True,
        random_seed=1,
    )
    return ts

###### Population Configuration
pop_configs= [
    msprime.PopulationConfiguration(initial_size=2600), # Den
    msprime.PopulationConfiguration(initial_size=2800), # Altai
    msprime.PopulationConfiguration(initial_size=20000), # AFR
]
###### Simulate Human to Neanderthal introgression
print("running human to neanderthal simulation")
dem_events= [
    ### Human Introgression from AFR1 to Neanderthal 120,000
    msprime.MassMigration(time = (250000/Gen), source=Altai, destination=AFR, proportion=0.05),
    ### Divergence of Neanderthal and Denisovan  at 473,000 years ago
    msprime.MassMigration(time = (473000/Gen), source=Den, destination=Altai, proportion=1),
    ### Divergence of Modern Human and Neanderthal at 765,000 years ago
    msprime.MassMigration(time = (765000/Gen), source=Altai, destination=AFR, proportion=1),
    ### Set ancestral population size
    msprime.PopulationParametersChange(time=(765000/Gen), initial_size=20000, growth_rate=0, population_id=Altai),
    msprime.PopulationParametersChange(time=(765000/Gen), initial_size=20000, growth_rate=0, population_id=Den)]
#### Print demography with DemographyDebugger
dd = msprime.DemographyDebugger(population_configurations=pop_configs, demographic_events=dem_events)
print(dd.print_history())
### run simulations
ts = archaic_introgression(pop_configs, dem_events)
### Output Files
# output tree sequences
ts.dump("HumantoNeanderthal.ts")
# output vcf file
with open("HumantoNeanderthal.vcf", "w") as vcf_file:
    ts.write_vcf(vcf_file, 2)
# output migration records for human to neanderthal introgression
with open("HumantoNeanderthal.migrations", "w") as migration_file:
    for migration in ts.migrations():
        if migration.time == (250000/29):
            migration_file.write(str(migration) + "\n")
##### Done
print("Done with Human to Neanderthal Simulation") 
#####

###### Simulate Human to Neanderthal introgression
print("running Neanderthal to human simulation")
dem_events= [
    ### Human Introgression from AFR1 to Neanderthal 120,000
    msprime.MassMigration(time = (50000/Gen), source=AFR, destination=Altai, proportion=0.05),
    ### Divergence of Neanderthal and Denisovan  at 473,000 years ago
    msprime.MassMigration(time = (473000/Gen), source=Den, destination=Altai, proportion=1),
    ### Divergence of Modern Human and Neanderthal at 765,000 years ago
    msprime.MassMigration(time = (765000/Gen), source=Altai, destination=AFR, proportion=1),
    ### Set ancestral population size
    msprime.PopulationParametersChange(time=(765000/Gen), initial_size=20000, growth_rate=0, population_id=Altai),
    msprime.PopulationParametersChange(time=(765000/Gen), initial_size=20000, growth_rate=0, population_id=Den)]
#### Print demography with DemographyDebugger
dd = msprime.DemographyDebugger(population_configurations=pop_configs, demographic_events=dem_events)
print(dd.print_history())
### run simulations
ts = archaic_introgression(pop_configs, dem_events)
### Output Files
# output tree sequences
ts.dump("NeanderthaltoHuman.ts")
# output vcf file
with open("NeanderthaltoHuman.vcf", "w") as vcf_file:
    ts.write_vcf(vcf_file, 2)
# output migration records for human to neanderthal introgression
with open("NeanderthaltoHuman.migrations", "w") as migration_file:
    for migration in ts.migrations():
        if migration.time == (50000/29):
            migration_file.write(str(migration) + "\n")
##### Done
print("Done with Neanderthal to Human Simulation")
#####
