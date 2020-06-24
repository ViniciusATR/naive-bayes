module Main where

import Data.List
import Data.Ord

--  (Classe, distribuiçoes, prior)
type Model = [(Int, [(Double -> Double)] , Double )]

--           Dataset         pontos individuais agrupados por classes
separate :: [([Double], Int)] -> [(Int, [[Double]])]
separate dt = map fixTuple grouped
  where
    fixTuple tpl = (head $ snd tpl , fst tpl)
    grouped = map unzip $ groupBy (\x y -> snd x == snd y) sorted
    sorted  = sortBy (comparing snd) dt

--            Matriz de feats  Vetor de médias
calculateMean :: [[Double]] -> [Double]
calculateMean dt = map (/fromIntegral (length dt)) sum
  where
    len  = length $ head dt 
    base = take len $ repeat 0.0
    sum  = foldr (\x y -> zipWith (+) x y) base dt

--          Matriz de feats  Vetor de desvio padrão
calculateSD :: [[Double]] -> [Double]
calculateSD dt = map sqrt variance
  where
    len  = length $ head dt
    base = take len $ repeat 0.0
    mean = calculateMean dt
    squaredSubMean = map (map (^2)) $ map (\x -> zipWith (-) x mean) dt
    sum  = foldr (\x y -> zipWith (+) x y) base dt
    variance = map (/fromIntegral (length dt) ) sum

--                  Dataframe por classe   (Classe, vetor de médias, vetor de DesvPad)
summarizeClasses :: [(Int, [[Double]])] -> [(Int, [Double], [Double])]
summarizeClasses dt = map summarizeClass dt
  where
    summarizeClass cls = (fst cls, calculateMean $ snd cls, calculateSD $ snd cls)

--                Media     DesvPad      Gaussiana
createGaussian :: Double -> Double -> (Double -> Double)
createGaussian m dp = (\x -> factor * exp (exponential x))
  where
    factor = 1.0 / (dp * sqrt (2 * pi))
    exponential x = (-0.5) * ((x - m)/dp)^2

--          (Classe, vetor de médias, vetor de DesvPad) (Classe, [distribuições gaussianas])
createGaussians :: [(Int, [Double], [Double])] -> [(Int, [(Double -> Double)])]
createGaussians dt = map gaussians dt
  where
    gaussians ( id , m , dp )  =  (id, zipWith (createGaussian) m dp)

--               Dataframe por classe   (classe, prior)
computePriors :: [(Int, [[Double]])] -> [(Int, Double)]
computePriors dt = map prior individualSums
  where
    individualSums = map (\(id, ls) -> (id, length ls)) dt
    total = foldr (+) 0 $ map snd individualSums
    prior (id, len) = (id, (fromIntegral len)/(fromIntegral total) )

--           Dataset treino        
trainNB :: [([Double], Int)] -> Model
trainNB dt = zipWith (\g p -> (fst g, snd g , snd p)) gaussians priors
  where
    grouped = separate dt
    gaussians = createGaussians $ summarizeClasses grouped
    priors = computePriors grouped

--                     feats       (classe, likelihoods, prior)            (classe, coef)
calculateClassCoef :: [Double] -> (Int , [(Double -> Double)] , Double) -> (Int , Double)
calculateClassCoef feats (cls, gss, p) = (cls, p * likelihood)
  where
    featProbs = zipWith (\x y -> x y) gss feats
    likelihood = foldr (*) 1.0 featProbs

predict :: Model -> [Double] -> Int
predict m tg = fst bestFit
  where
    clsCoefs = map (calculateClassCoef tg) m 
    bestFit = foldr (\x y -> if snd x >= snd y then x else y) (head clsCoefs) clsCoefs


main :: IO ()
main = do
  print $ map (predict (trainNB iris_train)) iris_test

--             Atributos, classe
iris_train :: [([Double], Int)]
iris_train = [
          ( [4.9, 3.1, 1.5, 0.2], 0 ),
          ( [5.0, 2.0, 3.5, 1.0], 1 ),
          ( [4.7, 3.2, 1.3, 0.2], 0 ),
          ( [4.8, 3.4, 1.9, 0.2], 0 ),
          ( [6.3, 2.7, 4.9, 1.8], 2 ),
          ( [5.0, 3.2, 1.2, 0.2], 0 ),
          ( [6.7, 3.3, 5.7, 2.1], 2 ),
          ( [6.2, 2.2, 4.5, 1.5], 1 ),
          ( [5.0, 3.4, 1.6, 0.4], 0 ),
          ( [4.7, 3.2, 1.6, 0.2], 0 ),
          ( [5.1, 3.8, 1.5, 0.3], 0 ),
          ( [4.5, 2.3, 1.3, 0.3], 0 ),
          ( [5.4, 3.9, 1.3, 0.4], 0 ),
          ( [5.4, 3.4, 1.7, 0.2], 0 ),
          ( [5.8, 2.7, 5.1, 1.9], 2 ),
          ( [5.4, 3.0, 4.5, 1.5], 1 ),
          ( [4.6, 3.2, 1.4, 0.2], 0 ),
          ( [6.7, 2.5, 5.8, 1.8], 2 ),
          ( [4.9, 3.0, 1.4, 0.2], 0 ),
          ( [5.0, 2.3, 3.3, 1.0], 1 ),
          ( [6.7, 3.3, 5.7, 2.5], 2 ),
          ( [7.2, 3.2, 6.0, 1.8], 2 ),
          ( [5.8, 2.6, 4.0, 1.2], 1 ),
          ( [6.7, 3.1, 4.7, 1.5], 1 ),
          ( [5.1, 3.8, 1.6, 0.2], 0 ),
          ( [7.7, 3.0, 6.1, 2.3], 2 ),
          ( [5.0, 3.4, 1.5, 0.2], 0 ),
          ( [6.7, 3.1, 4.4, 1.4], 1 ),
          ( [5.4, 3.7, 1.5, 0.2], 0 ),
          ( [6.4, 2.8, 5.6, 2.2], 2 ),
          ( [4.3, 3.0, 1.1, 0.1], 0 ),
          ( [5.7, 4.4, 1.5, 0.4], 0 ),
          ( [5.9, 3.0, 4.2, 1.5], 1 ),
          ( [6.1, 3.0, 4.6, 1.4], 1 ),
          ( [6.5, 3.0, 5.5, 1.8], 2 ),
          ( [5.2, 3.5, 1.5, 0.2], 0 ),
          ( [5.6, 2.5, 3.9, 1.1], 1 ),
          ( [7.7, 2.6, 6.9, 2.3], 2 ),
          ( [6.3, 3.4, 5.6, 2.4], 2 ),
          ( [6.2, 2.9, 4.3, 1.3], 1 ),
          ( [5.7, 2.9, 4.2, 1.3], 1 ),
          ( [5.0, 3.5, 1.6, 0.6], 0 ),
          ( [5.6, 2.9, 3.6, 1.3], 1 ),
          ( [6.0, 2.2, 5.0, 1.5], 2 ),
          ( [5.5, 2.6, 4.4, 1.2], 1 ),
          ( [4.6, 3.4, 1.4, 0.3], 0 ),
          ( [5.6, 3.0, 4.1, 1.3], 1 ),
          ( [5.1, 3.4, 1.5, 0.2], 0 ),
          ( [6.4, 2.9, 4.3, 1.3], 1 ),
          ( [6.8, 3.0, 5.5, 2.1], 2 ),
          ( [6.7, 3.0, 5.0, 1.7], 1 ),
          ( [6.5, 3.2, 5.1, 2.0], 2 ),
          ( [6.0, 3.4, 4.5, 1.6], 1 ),
          ( [4.9, 3.1, 1.5, 0.1], 0 ),
          ( [4.9, 2.5, 4.5, 1.7], 2 ),
          ( [6.9, 3.2, 5.7, 2.3], 2 ),
          ( [5.4, 3.4, 1.5, 0.4], 0 ),
          ( [5.5, 2.4, 3.8, 1.1], 1 ),
          ( [6.3, 3.3, 6.0, 2.5], 2 ),
          ( [5.0, 3.6, 1.4, 0.2], 0 ),
          ( [6.1, 3.0, 4.9, 1.8], 2 ),
          ( [6.5, 2.8, 4.6, 1.5], 1 ),
          ( [5.9, 3.0, 5.1, 1.8], 2 ),
          ( [6.3, 2.5, 4.9, 1.5], 1 ),
          ( [4.9, 3.6, 1.4, 0.1], 0 ),
          ( [6.1, 2.6, 5.6, 1.4], 2 ),
          ( [6.4, 3.2, 4.5, 1.5], 1 ),
          ( [7.1, 3.0, 5.9, 2.1], 2 ),
          ( [5.5, 3.5, 1.3, 0.2], 0 ),
          ( [6.4, 2.7, 5.3, 1.9], 2 ),
          ( [5.5, 2.3, 4.0, 1.3], 1 ),
          ( [6.9, 3.1, 5.4, 2.1], 2 ),
          ( [5.8, 2.7, 3.9, 1.2], 1 ),
          ( [5.8, 2.7, 5.1, 1.9], 2 ),
          ( [6.1, 2.8, 4.0, 1.3], 1 ),
          ( [5.9, 3.2, 4.8, 1.8], 1 ),
          ( [6.2, 3.4, 5.4, 2.3], 2 ),
          ( [5.7, 2.8, 4.1, 1.3], 1 ),
          ( [6.6, 3.0, 4.4, 1.4], 1 ),
          ( [5.8, 2.8, 5.1, 2.4], 2 ),
          ( [6.0, 2.2, 4.0, 1.0], 1 ),
          ( [5.8, 2.7, 4.1, 1.0], 1 ),
          ( [4.6, 3.6, 1.0, 0.2], 0 ),
          ( [6.7, 3.1, 5.6, 2.4], 2 ),
          ( [5.1, 3.3, 1.7, 0.5], 0 ),
          ( [6.8, 2.8, 4.8, 1.4], 1 ),
          ( [5.3, 3.7, 1.5, 0.2], 0 ),
          ( [5.2, 2.7, 3.9, 1.4], 1 ),
          ( [6.1, 2.8, 4.7, 1.2], 1 ),
          ( [5.5, 2.5, 4.0, 1.3], 1 ),
          ( [5.5, 2.4, 3.7, 1.0], 1 ),
          ( [5.1, 3.5, 1.4, 0.3], 0 ),
          ( [6.7, 3.0, 5.2, 2.3], 2 ),
          ( [6.0, 3.0, 4.8, 1.8], 2 ),
          ( [5.7, 3.0, 4.2, 1.2], 1 ),
          ( [5.1, 2.5, 3.0, 1.1], 1 ),
          ( [5.7, 2.6, 3.5, 1.0], 1 ),
          ( [4.6, 3.1, 1.5, 0.2], 0 ),
          ( [5.2, 3.4, 1.4, 0.2], 0 ),
          ( [6.9, 3.1, 5.1, 2.3], 2 ),
          ( [5.6, 2.8, 4.9, 2.0], 2 ),
          ( [5.5, 4.2, 1.4, 0.2], 0 ),
          ( [4.8, 3.0, 1.4, 0.3], 0 ),
          ( [4.4, 3.2, 1.3, 0.2], 0 ),
          ( [6.4, 3.2, 5.3, 2.3], 2 ),
          ( [5.0, 3.5, 1.3, 0.3], 0 ),
          ( [5.2, 4.1, 1.5, 0.1], 0 ),
          ( [6.3, 2.5, 5.0, 1.9], 2 ),
          ( [6.8, 3.2, 5.9, 2.3], 2 ),
          ( [5.6, 3.0, 4.5, 1.5], 1 ),
          ( [5.0, 3.3, 1.4, 0.2], 0 ),
          ( [4.4, 2.9, 1.4, 0.2], 0 ),
          ( [4.8, 3.1, 1.6, 0.2], 0 ),
          ( [7.7, 3.8, 6.7, 2.2], 2 ),
          ( [6.3, 3.3, 4.7, 1.6], 1 ),
          ( [5.1, 3.7, 1.5, 0.4], 0 ),
          ( [5.1, 3.5, 1.4, 0.2], 0 ),
          ( [7.9, 3.8, 6.4, 2.0], 2 ),
          ( [6.9, 3.1, 4.9, 1.5], 1 ),
          ( [6.2, 2.8, 4.8, 1.8], 2 ),
          ( [4.4, 3.0, 1.3, 0.2], 0 ),
          ( [5.1, 3.8, 1.9, 0.4], 0 ),
          ( [6.5, 3.0, 5.2, 2.0], 2 ),
          ( [4.9, 2.4, 3.3, 1.0], 1 ),
          ( [5.7, 2.8, 4.5, 1.3], 1 ),
          ( [5.6, 2.7, 4.2, 1.3], 1 ),
          ( [7.2, 3.6, 6.1, 2.5], 2 ),
          ( [6.3, 2.9, 5.6, 1.8], 2 ),
          ( [6.6, 2.9, 4.6, 1.3], 1 ),
          ( [6.4, 3.1, 5.5, 1.8], 2 ),
          ( [7.0, 3.2, 4.7, 1.4], 1 ),
          ( [6.3, 2.3, 4.4, 1.3], 1 ),
          ( [6.5, 3.0, 5.8, 2.2], 2 ),
          ( [7.2, 3.0, 5.8, 1.6], 2 ),
          ( [7.7, 2.8, 6.7, 2.0], 2 )
          ]


iris_test = [[6.4, 2.8, 5.6, 2.1],
        [5.7, 3.8, 1.7, 0.3],
        [7.4, 2.8, 6.1, 1.9],
        [7.6, 3.0, 6.6, 2.1],
        [7.3, 2.9, 6.3, 1.8],
        [6.0, 2.9, 4.5, 1.5],
        [6.0, 2.7, 5.1, 1.6],
        [5.8, 4.0, 1.2, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [6.3, 2.8, 5.1, 1.5],
        [5.0, 3.0, 1.6, 0.2],
        [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1],
        [6.1, 2.9, 4.7, 1.4],
        [5.7, 2.5, 5.0, 2.0]]