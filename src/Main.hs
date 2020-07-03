module Main where

import Data.List
import Data.List.Extra (groupSortBy)
import Data.List.Split (splitOn)
import Data.Ord
import Control.Lens

--  (Classe, distribuiçoes, prior)
type Model = [(Int, [Double -> Double] , Double )]

--           Dataset         pontos individuais agrupados por classes
separate :: [([Double], Int)] -> [(Int, [[Double]])]
separate dt = map fixTuple grouped
  where
    fixTuple tpl = (head $ snd tpl , transpose $ fst tpl)
    grouped = map unzip $ groupSortBy (\x y -> snd x `compare` snd y) dt


mean :: [Double] -> Double
mean xs = sum / len
  where
    (sum, len) = foldr (\x y -> (x + fst y, snd y + 1.0)) (0.0, 0.0) xs


sd :: [Double] -> Double
sd xs = sqrt variance
  where
    m = mean xs
    variance = mean $ map (\x -> (x-m)^2) xs


--                  Dataframe por classe   (Classe, vetor de médias, vetor de DesvPad)
summarizeClasses :: [(Int, [[Double]])] -> [(Int, [Double], [Double])]
summarizeClasses  = map summarizeClass
  where
    summarizeClass cls = (fst cls, map mean $ snd cls, map sd $ snd cls)

--                Media     DesvPad      Gaussiana
createGaussian :: Double -> Double -> (Double -> Double)
createGaussian m dp = \x -> factor * exp (exponential x)
  where
    factor = 1.0 / (dp * sqrt (2 * pi))
    exponential x = (-0.5) * ((x - m)/dp)^2

--          (Classe, vetor de médias, vetor de DesvPad) (Classe, [distribuições gaussianas])
createGaussians :: [(Int, [Double], [Double])] -> [(Int, [Double -> Double])]
createGaussians = map gaussians
  where
    gaussians ( id , m , dp )  =  (id, zipWith createGaussian m dp)

--               Dataframe por classe   (classe, prior)
computePriors :: [(Int, [[Double]])] -> [(Int, Double)]
computePriors dt = map prior individualSums
  where
    individualSums = map (\(id, ls) -> (id, length ls)) dt
    total = foldr ((+).snd) 0 individualSums
    prior (id, len) = (id, fromIntegral len/fromIntegral total )

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
      let ds = parseStringDataset r
      print "Digite o path do target"
      ptg <- getLine
      r   <- readFile ptg
      let tg = parseTarget r
      print $ map (predict (trainNB ds)) tg
