module Main where

import Data.List
import Data.List.Split (splitOn)
import Data.Ord
import Control.Lens

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

parseStringDataset :: String -> [([Double], Int)]
parseStringDataset s = map transform tpl
  where
    ls  = map (splitOn ",") $ lines s
    tpl = map (\x -> ( x^?!_init , last x)) ls
    transform tp = (map read (fst tp) :: [Double] , read (snd tp) :: Int)

parseTarget :: String -> [[Double]]
parseTarget s = map transform ls
  where
    ls = map (splitOn ",") $ lines s
    transform l = map read l :: [Double]

main :: IO ()
main = do
      print "Digite o path do dataset"
      pds <- getLine
      r   <- readFile pds
      ds  <- return $ parseStringDataset r
      print "Digite o path do target"
      ptg <- getLine
      r   <- readFile ptg
      tg  <- return $ parseTarget r
      print $ map (predict (trainNB ds)) tg


--  print $ map (predict (trainNB iris_train)) iris_test
