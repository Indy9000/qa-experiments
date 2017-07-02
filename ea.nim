import os, unicode, strutils, streams, tables, parsecsv, algorithm, random

const
  GenerationCount = 500
  PopulationCount = 100
  ChromosomeLength = 10
  MutationProbability = 0.2
  CrossoverProbability = 0.5
type
  TChromosome = array[ChromosomeLength,float]
  TPopulation = array[PopulationCount,TChromosome]
  TFitness = array[PopulationCount,float]
  TAgent = tuple[epochCount:int, batchSize:int, 
                  filterCount:int, kernelSize:int, 
                  dropout:float,denseSize:int]
# --------------------------------------------------------------------------------------------
proc get_random_vector[T]():array[ChromosomeLength, T] =
  var vector:array[ChromosomeLength, T]
  for i in 0..ChromosomeLength-1:
    vector[i] = random((T)1.0)
  return vector
# --------------------------------------------------------------------------------------------
proc max_array_index[T](length:int, values:openArray[T]):int = 
  var 
    max = int.low
    max_i = -1

  for i in 0..length-1:
    if max < values[i]: 
      max = values[i]
      max_i = i

  return max_i
# --------------------------------------------------------------------------------------------
proc mutate(winner:TChromosome, loser:var TChromosome, p_mu:float, p_xo:float)=
  let mutation_probabilities = get_random_vector[float]()
  let crossover_probabilities = get_random_vector[float]()
  for i in 0..len(winner)-1:
    if crossover_probabilities[i] >= p_xo:
      loser[i] = winner[i]

    if mutation_probabilities[i] >= p_mu:
      let d = 0.1 # mutation delta
      let dmu = mutation_probabilities[i] * d - d/2.0 #scale random mutation and shift range
      let tv = loser[i] + dmu # random mutation applied
      #clip to the 0..1 range and apply
      loser[i] = if tv < 0.0: 0.0 else: (if tv>1.0: 1.0 else: tv)
# --------------------------------------------------------------------------------------------
proc tournament(population:var TPopulation, fitness:TFitness, p_mu:float, p_xo:float) =
  # divide the population into two and pit them against each other
  # based on their fitness
  var indices:array[PopulationCount,int]
  #initialise indices
  for i in 0..PopulationCount-1:
    indices[i] = i;
  #inplace random shuffle
  shuffle(indices) 

  var i = 0
  while i < PopulationCount-1:
    if fitness[i] > fitness[i+1]:
      mutate(population[i], population[i+1], p_mu, p_xo)
    else:
      mutate(population[i+1], population[i], p_mu, p_xo)
    i = i+2
# --------------------------------------------------------------------------------------------
proc linear_scale(value:float, rLo, rHi: float):int = 
  # Assumes value is normalized to 0..1
  return int(value * (rHi - rLo) + rLo)
# --------------------------------------------------------------------------------------------
proc map_chromosome(chromosome:TChromosome):TAgent =
  var agent:TAgent

  agent = (
    epochCount:linear_scale(chromosome[0], 10,100), 
    batchSize:linear_scale(chromosome[1],5,300),
    filterCount:linear_scale(chromosome[2],2,100),
    kernelSize:linear_scale(chromosome[3],2,6),
    dropout:chromosome[4],
    denseSize:linear_scale(chromosome[5],5,50)
  )
  return agent
# --------------------------------------------------------------------------------------------
proc parse_output(filename:string) : tuple[tl:seq[float],ta:seq[float],
                                          vl:seq[float],va:seq[float]] =
  var tr_loss = @[1.0]
  var tr_acc  = @[0.0]
  var va_loss = @[1.0]
  var va_acc  = @[0.0]

  for line in lines filename:
    if "val_loss" in line:
      let li = line.strip()
      let token="[==============================]"
      let k = li.find(token)
      if k > -1:
        let ll = li[(k+len(token)) .. ^1]
        let tokens = splitWhitespace(ll)
        echo "tr_loss ",tokens[4]," tr_acc ",tokens[7]," va_loss ",tokens[10]," va_acc ",tokens[13]
        tr_loss.add(parseFloat(tokens[4]))
        tr_acc.add(parseFloat(tokens[7]))
        va_loss.add(parseFloat(tokens[10]))
        va_acc.add(parseFloat(tokens[13]))

  return (tr_loss,tr_acc,va_loss,va_acc)

# --------------------------------------------------------------------------------------------
proc compute_performance(tl:seq[float], ta:seq[float],
                          vl:seq[float], va:seq[float]):float =
  for vv in 0..len(ta)-1:
    if ta[vv] >= 0.98:
      let tl = if tl[vv] > 1.0: 1.0 else: (1.0-tl[vv])
      let vl = if vl[vv] > 1.0: 1.0 else: (1.0-vl[vv])
      return ta[vv] * tl * va[vv] * vl
  let ee = len(ta)-1
  return ta[ee] * (1.0 - tl[ee]) * va[ee] * (1.0 - vl[ee])

# --------------------------------------------------------------------------------------------
proc execute_agent(agent:TAgent, generation:int, popIndex:int):float=
  echo "Excuting agent ",generation,popIndex
  let filename = "output-" & $(generation) & $(popIndex)
  #execute python script with parameters extracted from agent object
  let retval = execShellCmd("runner.sh " & $(agent.epochCount) & " " &
                              $(agent.batchSize) & " "  & 
                              $(agent.filterCount) & " "  &
                              $(agent.kernelSize) & " " &
                              $(agent.dropout) & " "  &
                              $(agent.denseSize) & " |" & filename
              )

  #once finished, parse output
  let (tl,ta,vl,va) = parse_output(filename)
  let fitness = compute_performance(tl,ta,vl,va)
  echo "fitness ",generation,popIndex,fitness
  return fitness

# --------------------------------------------------------------------------------------------
proc evaluate_population(population:TPopulation, generation:int):
            tuple[fitness:TFitness,max_fitness:float,best_agent:TAgent] = 
  var
    i = 0
    fitness:TFitness
    max_fitness:float
    best_agent:TAgent
  # evaluate each agent
  for c in population:
    # express each chromosome
    let agent = map_chromosome(c)   
    # execute each agent
    fitness[i] = execute_agent(agent, generation, i)
    
    if max_fitness <= fitness[i]:
      max_fitness = fitness[i]
      best_agent = agent
    
    i += 1
  
  return (fitness, max_fitness, best_agent)

# --------------------------------------------------------------------------------------------
proc main() =
  if paramCount() != 1:
    quit("synopsis: " & getAppFilename() & " exp-output-filename")

  let filename = paramStr(1)
  
  var
    population:TPopulation
    fitness:TFitness

  # generate a random population
  for i in 0..PopulationCount-1:
    population[i] = get_random_vector[float]()
    fitness[i] = 0.0

  for g in 0..GenerationCount-1:
    let (fitness, max_fitness, best_agent) = evaluate_population(population, g)
    echo "Best fitness =", max_fitness, best_agent
    tournament(population, fitness, MutationProbability, CrossoverProbability)
# --------------------------------------------------------------------------------------------
main()
