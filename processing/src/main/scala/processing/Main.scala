package processing

import fs2.{io, text, Task, Stream}
import java.nio.file.{Paths, Path, Files}

object Main extends App {
  val englishDir   = Paths.get("../data/Universal Dependencies 1.3/ud-treebanks-v1.3/UD_English")
  val processedDir = Paths.get("../data/processed/")
  
  val enDev     = englishDir.resolve("en-ud-dev.conllu")
  val enDevProc = processedDir.resolve("en-ud.dev.txt")

  def converter(from: Path, to: Path): Stream[Task, Unit] =
    io.file.readAll[Task](from, 4096)
      .through(text.utf8Decode)
      .through(text.lines)
      .map { line =>
        val elems = line.split("\\s")
        if (elems.length < 2) None
        else Some(elems(1))
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
      .through(text.utf8Encode)
      .through(io.file.writeAll(to))

  // if (!Files.exists(enDevProc)) Files.createFile(enDevProc)
  converter(enDev, enDevProc).run.unsafeRun
}
