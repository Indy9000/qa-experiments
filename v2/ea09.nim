import os, times, unicode, strutils, streams, tables, parsecsv, algorithm, random, threadpool


const
  ChromosomeLength = 9
type
  TChromosome = tuple[Chromo:array[ChromosomeLength,float], fitness:float, data:string]
  TAgent = tuple[ filterCount:int, kernelSize:int, poolingWindow:int, 
                  denseSize1:int, denseSize2:int,
                  dropout1:float, dropout2:float, batchSize:int, epochCount:int]
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
      let d = 0.01 # mutation delta
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
  echo "shuffled indices=",$(indices)

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
    filterCount:  linear_scale(c.Chromo[0],2,50),
    kernelSize:   linear_scale(c.Chromo[1],2,6),
    poolingWindow:linear_scale(c.Chromo[2],5,50),
    denseSize1:    linear_scale(c.Chromo[3],10,50),
    denseSize2:    linear_scale(c.Chromo[4],10,50),
    dropout1:                   c.Chromo[5],
    dropout2:                   c.Chromo[6],
    batchSize:    linear_scale(c.Chromo[7],4,64),
    epochCount:   linear_scale(c.Chromo[8],10,50) 
  )
  return agent
# --------------------------------------------------------------------------------------------
proc parse_output(filename:string) : (float,float,float,float,float) =
    #Line to look for an parse
    #Metrics-val: precision= 1.0 ,recall= 0.136363636364 ,f1= 0.24 ,avg_prec= 0.598404511713
    #Val Mean avg Prec = 0.156140735555

    let token1="Metrics-val:"
    let token2="Val Mean avg Prec = "
    var precision,recall,f1,avg_prec,mean_avg_prec : float
    for line in lines filename:
        let k1 = line.find(token1)
        if k1 > -1:
            let ll = line[(k1+len(token1)) .. ^1]
            let split_tokens = splitWhitespace(ll)
            precision = parseFloat(split_tokens[1])
            recall    = parseFloat(split_tokens[3])
            f1        = parseFloat(split_tokens[5])
            avg_prec  = parseFloat(split_tokens[7])

        let k2 = line.find(token2)
        if k2 > -1:
            let ll = line[(k2+len(token2)) .. ^1]
            mean_avg_prec = parseFloat(ll)
    return (precision,recall,f1,avg_prec,mean_avg_prec)

# --------------------------------------------------------------------------------------------
proc parse_output1(filename:string) : (float,float,float,float,float,float) =
    #Line to look for an parse
    #26s - loss: 0.3458 - acc: 0.8907 - val_loss: 0.3320 - val_acc: 0.896
    let token1="s - loss: "
    let token2="Val Mean avg Prec = "
    let token3="Test Mean avg Prec = "
    var tr_loss,tr_acc,val_loss,val_acc : float
    var te_map,val_map : float

    for line in lines filename:
        let k1 = line.find(token1)
        if k1 > -1:
            let ll = line[(k1+len(token1)) .. ^1]
            let split_tokens = splitWhitespace(ll)
            tr_loss  = parseFloat(split_tokens[0])
            tr_acc   = parseFloat(split_tokens[3])
            val_loss = parseFloat(split_tokens[6])
            val_acc  = parseFloat(split_tokens[9])

        let k2 = line.find(token2)
        if k2 > -1:
            let ll = line[(k2+len(token2)) .. ^1]
            val_map = parseFloat(strip(ll))
        
        let k3 = line.find(token3)
        if k3 > -1:
            let ll = line[(k3+len(token3)) .. ^1]
            te_map = parseFloat(strip(ll))
    return (tr_loss,tr_acc,val_loss,val_acc,val_map,te_map)

# --------------------------------------------------------------------------------------------
proc compute_performance(tr_l:float, tr_a:float, va_l:float, va_a:float, va_map:float, te_map:float):(float,string) =
  let data = "tr_loss="   & $(tr_l) & ", tr_acc=" & $(tr_a) &
             ", va_loss=" & $(va_l) & ", va_acc=" & $(va_a) &
             ", va_map="  & $(va_map) & ", te_map=" & $(te_map)
  return (va_map, data)

# --------------------------------------------------------------------------------------------
proc execute_agent(agent:TAgent, generation:int, popIndex:int,output_dir:string):float=
  echo "Excuting agent " & intToStr(generation,4) & "-" & intToStr(popIndex,4)
  let filename = output_dir &
                  "/output9-" &
                  intToStr(generation,4) &
                  "-" &
                  intToStr(popIndex,4) & ".txt"
  #execute python script with parameters extracted from agent object
  let retval = execShellCmd("python exp-09.py " &
                              $(agent.filterCount) & " "  &
                              $(agent.kernelSize) & " " &
                              $(agent.poolingWindow) & " " &
                              $(agent.denseSize1) & " " &
                              $(agent.denseSize2) & " " &
                              $(agent.dropout1) & " "  &
                              $(agent.dropout2) & " "  &
                              $(agent.batchSize) & " "  &
                              $(agent.epochCount) & " |tr -d '\b\r' |sed '/ETA:/d' >" & filename
              )

  #once finished, parse output
  let (tr_l,tr_a,va_l,va_a,va_map,te_map) = parse_output1(filename)
  let (fitness,data) = compute_performance(tr_l,tr_a,va_l,va_a,va_map,te_map)
  echo "Generation ",generation," ",popIndex," Fitness= ",fitness," ",data
  return fitness

# --------------------------------------------------------------------------------------------
{.experimental.}
proc evaluate_population(population:var openArray[TChromosome],
                                generation:int,
                                output_dir:string):(float, TAgent, int) =

  # evaluate each agent
  parallel:
    for i in 0..len(population)-1:
      # express each chromosome
      # execute each agent
      population[i].fitness = spawn execute_agent(map_chromosome(population[i]), generation, i,
                                                  output_dir)
  
  var
    max_fitness:float = -1.7976931348623157e+308
    max_index:int = -1
  for i in 0..len(population)-1:
    if max_fitness <= population[i].fitness:
       max_fitness = population[i].fitness
       max_index = i

  let best_agent = map_chromosome(population[max_index])
  
  return (max_fitness, best_agent, max_index)

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
    let (max_fitness, best_agent, max_index) = evaluate_population(population, g, output_dir)
    echo "[",getGmTime(getTime()), "]Best fitness =", max_fitness,
          " output9-", intToStr(g,4),"-", intToStr(max_index,4), ".txt ",
          best_agent
    tournament(population, MutationProbability, CrossoverProbability)
# --------------------------------------------------------------------------------------------
main()
