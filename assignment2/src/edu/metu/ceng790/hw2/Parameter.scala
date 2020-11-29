package edu.metu.ceng790.hw2

class Parameter(Rank: Int, Iteration: Int, Lambda: Double) extends Serializable {
  val rank: Int = Rank
  val iteration: Int = Iteration
  val lambda: Double = Lambda

  override def toString: String = "Parameter(Rank: %d, Iteration: %d, Lambda: %.2f)".format(rank, iteration, lambda)
  def getShortDescription: String = "Rank%dIteration%dLambda%.2f".format(rank, iteration, lambda).replace(",", ".")
}
