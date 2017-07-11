import os, unicode, strutils, streams, tables, parsecsv, algorithm, random, threadpool


const
  ChromosomeLength = 6
type
  TChromosome = tuple[Chromo:array[ChromosomeLength,float], fitness:float]
  TAgent = tuple[ filterCount:int, kernelSize:int, poolingWindow:int,
                  dropout:float, batchSize:int, epochCount:int]
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
  for i in 0..ChromosomeLength-1:
    if crossover_probabilities[i] >= p_xo:
      loser.Chromo[i] = winner.Chromo[i]

    if mutation_probabilities[i] >= p_mu:
      let d = 0.1 # mutation delta
      let dmu = mutation_probabilities[i] * d - d/2.0 #scale random mutation and shift range
      let tv = loser.Chromo[i] + dmu # random mutation applied
      #clip to the 0..1 range and apply
      loser.Chromo[i] = if tv < 0.0: 0.0 else: (if tv>1.0: 1.0 else: tv)
# --------------------------------------------------------------------------------------------
proc tournament(population:var openArray[TChromosome], p_mu:float, p_xo:float) =
  # divide the population into two and pit them against each other
  # based on their fitness
 # var indices:array[PopulationCount,int]
  let popSize = len(population)
  var indices = newSeq[int](popSize)
  #initialise indices
  for i in 0..popSize-1:
    indices[i] = i;
  #inplace random shuffle
  shuffle(indices) 

  var i = 0
  while i < popSize-1:
    if population[i].fitness > population[i+1].fitness:
      mutate(population[i], population[i+1], p_mu, p_xo)
    else:
      mutate(population[i+1], population[i], p_mu, p_xo)
    i = i+2
# --------------------------------------------------------------------------------------------
proc linear_scale(value:float, rLo, rHi: float):int = 
  # Assumes value is normalized to 0..1
  return int(value * (rHi - rLo) + rLo)
# --------------------------------------------------------------------------------------------
proc map_chromosome(c:TChromosome):TAgent =
  var agent:TAgent

  agent = (
    filterCount:  linear_scale(c.Chromo[0],2,100),
    kernelSize:   linear_scale(c.Chromo[1],2,6),
    poolingWindow:linear_scale(c.Chromo[2],5,50),
    dropout:                   c.Chromo[3],
    batchSize:    linear_scale(c.Chromo[4],5,300),
    epochCount:   linear_scale(c.Chromo[5],10,100) 
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
proc execute_agent(agent:TAgent, generation:int, popIndex:int,output_dir:string):float=
  echo "Excuting agent " & intToStr(generation,4) & "-" & intToStr(popIndex,4)
  let filename = output_dir & 
                  "/output-" & 
                  intToStr(generation,4) & 
                  "-" & 
                  intToStr(popIndex,4) & ".txt"
  #execute python script with parameters extracted from agent object
  let retval = execShellCmd("python exp-02.py " & 
                              $(agent.filterCount) & " "  &
                              $(agent.kernelSize) & " " &
                              $(agent.poolingWindow) & " " &
                              $(agent.dropout) & " "  &
                              $(agent.batchSize) & " "  & 
                              $(agent.epochCount) & " > " & filename
              )

  #once finished, parse output
  let (tl,ta,vl,va) = parse_output(filename)
  let fitness = compute_performance(tl,ta,vl,va)
  echo "fitness ",generation,popIndex,fitness
  return fitness

# --------------------------------------------------------------------------------------------
{.experimental.}
proc evaluate_population(population:var openArray[TChromosome], generation:int, 
                         output_dir:string):
                                                  tuple[max_fitness:float,best_agent:TAgent] = 

  # evaluate each agent
  parallel:
    for i in 0..len(population)-1:
      # express each chromosome
      # execute each agent
      population[i].fitness = spawn execute_agent(map_chromosome(population[i]), generation, i, 
                                                  output_dir)
  
  var
    max_fitness:float
    max_index:int
  for i in 0..len(population)-1:
    if max_fitness <= population[i].fitness:
      max_fitness = population[i].fitness
      max_index = i

  let best_agent = map_chromosome(population[max_index])
  
  return (max_fitness, best_agent)

# --------------------------------------------------------------------------------------------
proc main() =
  if paramCount() != 5:
    quit("synopsis: " & getAppFilename() & " generation-count population-size mup xop output-dir")

  let GenerationCount = parseInt(paramStr(1))
  let PopulationSize = parseInt(paramStr(2))
  let MutationProbability = parseFloat(paramStr(3))
  let CrossoverProbability = parseFloat(paramStr(4))
  let output_dir = paramStr(5)

  var
    population = newSeq[TChromosome](PopulationSize)

  # generate a random population
  for i in 0..PopulationSize-1:
    population[i].Chromo = get_random_vector[float]()
    population[i].fitness = 0.0

  for g in 0..GenerationCount-1:
    let (max_fitness, best_agent) = evaluate_population(population, g, output_dir)
    echo "Best fitness =", max_fitness, best_agent
    tournament(population, MutationProbability, CrossoverProbability)
# --------------------------------------------------------------------------------------------
main()
