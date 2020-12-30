package edu.metu.ceng790.hw3

import org.apache.hadoop.mapred.InvalidInputException

import scala.io.StdIn.readLine
import org.apache.spark.ml.PipelineModel

object CreditApp {

  val Q_CONTINUE = "Do you want to continue credit risk assessment?"
  val A_CONTINUE = Array("Y", "N")

  val Q_CHECKING_ACCOUNT = "Status of existing checking account:\n" +
    "\t0: ... < 0 DM\n" +
    "\t1: 0 <= ... <= 200 DM\n" +
    "\t2: ... >= 200 DM/salary assignments for at least 1 year\n" +
    "\t3: no checking account\n"
  val A_CHECKING_ACCOUNT = Range(0,4).toArray

  val Q_DURATION = "Duration in month:"
  val A_DURATION = Array.empty[Int]

  def getStringAnswer(question: String, possibleInputs: Array[String]): String = {
    while (true) {
      val q = s"$question [${possibleInputs.mkString("/")}]: "
      val answer = readLine(q)

      if (possibleInputs.map(r => r.toUpperCase).contains(answer.toUpperCase)) return answer.toUpperCase
      else println(s"""Invalid Answer: "$answer"""")
    }
    return ""
  }

  def getIntAnswer(question: String, possibleInputs: Array[Int]): Int = {
    while (true) {
      val answer = readLine(question)
      try {
        val intAnswer = answer.toInt

        if (possibleInputs.isEmpty)
          return intAnswer
        else if (possibleInputs.nonEmpty && possibleInputs.contains(intAnswer))
          return intAnswer
        else
          println(s"""Invalid Answer: "$intAnswer"""")
      } catch {
        case e: NumberFormatException => println(s"""Invalid Answer: "$answer"""")
      }
    }
    return -1
  }

  def main(args: Array[String]) {

    try {
      val model = PipelineModel.load(Credit.MODEL_PATH)

      println("Welcome to the Credit Risk Assessment Tool")

      var keepContinuing = true
      while (keepContinuing) {
        val continue = getStringAnswer(Q_CONTINUE, A_CONTINUE)
        println()

        if (continue.equalsIgnoreCase("Y")) {
          val checkingAccount = getIntAnswer(Q_CHECKING_ACCOUNT, A_CHECKING_ACCOUNT)
          println()
          val duration = getIntAnswer(Q_DURATION, A_DURATION)
          println()
        } else {
          keepContinuing = false
        }
      }
    } catch {
      case e: InvalidInputException => println(s"""Model cannot be loaded from \"${Credit.MODEL_PATH}\" directory.""")
    }
  }
}
