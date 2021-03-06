package processing

import fs2.{io, text, Task, Stream}
import java.nio.file.{Paths, Path, Files}

object Main extends App {
  val englishDir   = Paths.get("../data/Universal Dependencies 1.3/ud-treebanks-v1.3/UD_English")
  val processedDir = Paths.get("../data/processed/")
  
  val enDev   = englishDir  .resolve("en-ud-dev.conllu")
  val enTest  = englishDir  .resolve("en-ud-test.conllu")
  val enTrain = englishDir  .resolve("en-ud-train.conllu")
  val enProc  = processedDir.resolve("en-ud.txt")

  val allowedCharsSet = Set('.', ',', ' ')

  def allowedChars(c: Char): Boolean =
    c.isLetter || c.isDigit || allowedCharsSet(c)

  def converter(from: Path): Stream[Task, String] =
    io.file.readAll[Task](from, 4096)
      .through(text.utf8Decode)
      .through(text.lines)
      .map { line =>
        val elems = line.split("\\s")
        if (elems.length < 2) None
        else Some(elems(1).toLowerCase.filter(allowedChars))
      }
      .split(_.isEmpty)
      .map { _
        .map(_.get)
        .foldLeft("") { case (accum, w) =>
          if (w.length == 1 && w.forall(!_.isLetter)) accum + w else accum + " " + w
        }
        .trim
      }
      .intersperse("\n")

  def write(to: Path): Stream[Task, String] => Stream[Task, Unit] = _
    .through(text.utf8Encode)
    .through(io.file.writeAll(to))

  implicit val s = fs2.Strategy.fromFixedDaemonPool(8, threadName = "worker")
  converter(enTrain)
    .merge(converter(enDev))
    .merge(converter(enTest))
    .through(write(enProc))
    .run.unsafeRun
}
