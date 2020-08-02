{-# LANGUAGE BangPatterns              #-}
{-# LANGUAGE ExistentialQuantification #-}

module Main where

import Prelude hiding (sum)
import Data.Monoid hiding (Sum)
import Control.Applicative
import Data.List hiding (sum)
import Data.List.Extra (groupSortBy)
import Data.List.Split (splitOn)
import Data.Ord
import Data.Tuple
import Control.Lens hiding(Fold)

import qualified Data.Foldable as F
import qualified Data.Map.Strict as Map
import qualified Data.Bifunctor as B

---------------
-- Fold code --
---------------

data Fold i o = forall m . Monoid m => Fold (i -> m) (m -> o)

instance Functor (Fold i) where
  fmap f (Fold liftContext process) = Fold liftContext ( f . process )

instance Applicative (Fold i) where
  pure f = Fold (\_ -> ()) (\_ -> f)

  Fold liftContextF processF <*> Fold liftContextX processX = Fold liftContext process
    where
      liftContext i = (liftContextF i, liftContextX i)

      process ( contextualizedF, contextualizedX ) = processF contextualizedF (processX contextualizedX)

data Count = Count !Int deriving(Show)

instance Semigroup Count where
  Count ca <> Count cb = Count ( ca + cb )

instance Monoid Count where
  mempty = Count 0

data Sum a = Sum !a deriving(Show)

instance Num a => Semigroup (Sum a) where
  Sum xa <> Sum xb = Sum ( xa + xb )

instance Num a => Monoid (Sum a) where
  mempty = Sum 0

newtype GroupBy k a = GroupBy { groupMap :: Map.Map k a } deriving(Show)

instance (Monoid a, Ord k) => Semigroup (GroupBy k a) where
  GroupBy xa <> GroupBy xb = GroupBy (Map.unionWith (<>) xa xb)

instance (Monoid a, Ord k) => Monoid (GroupBy k a) where
  mempty = GroupBy (Map.empty)

fold :: Fold i o -> [i] -> o
fold (Fold liftContext process ) is = process (reduce (map liftContext is))
  where reduce = F.foldl'(<>) mempty

count = Fold liftContext process
  where
    liftContext _ = Count 1
    process (Count c) = c

groupByF :: Ord a => Fold i o -> Fold (a, i) [(a, o)]
groupByF (Fold l p) = Fold liftContext process
  where
    liftContext (k,v) = GroupBy ( uncurry Map.singleton $ B.second l (k,v))
    process (GroupBy m) = map (B.second p) $ Map.toList m

sum :: Fractional a => Fold a a
sum = Fold liftContext process
  where
    liftContext x = Sum x
    process (Sum x) = x

sqsum :: Fractional a => Fold a a
sqsum = Fold liftContext process
  where
    liftContext x = Sum ( x ^ 2 )
    process (Sum x) = x

average :: Fractional a => Fold a a
average = (/) <$> sum <*> ( fmap fromIntegral count )

variance :: Fractional a => Fold a a
variance = (-) <$> avgSqrs <*> (fmap (^2) average)
  where avgSqrs = (/) <$> sqsum <*> ( fmap fromIntegral count )

combine :: Fold i a -> Fold i b -> Fold i (a, b)
combine = liftA2 (,)

stats = combine average (fmap sqrt variance)

freq :: (Ord o, Fractional a) => Fold (o,a) [(o,a)]
freq = extractFreq <$> countTotalPartial
  where
    integralPartial p = map (B.second fromIntegral) p
    extractFreq (c,p) = map (B.second (/fromIntegral c)) (integralPartial p)
    countTotalPartial = combine count (groupByF count)
----------------------
-- Naive Bayes code --
----------------------


--  (Classe, distribuiçoes, prior)
type Model = [(Int, [Double -> Double] , Double )]

--           Dataset         valores de colunas agrupados por classes
separate :: [([Double], Int)] -> [(Int, [[Double]])]
separate dt = map fixTuple grouped
  where
    fixTuple tpl = swap $ B.bimap transpose head tpl
    grouped = map unzip $ groupSortBy (\x y -> snd x `compare` snd y) dt --comparewithsecond procurar


mean :: [Double] -> Double
mean xs = fold average xs


sd :: [Double] -> Double
sd xs = fold (fmap sqrt variance) xs

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
    individualSums = map (B.second length) dt
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
calculateClassCoef :: [Double] -> (Int , [Double -> Double] , Double) -> (Int , Double)
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
