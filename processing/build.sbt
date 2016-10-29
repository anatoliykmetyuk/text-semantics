val ScalaVer = "2.11.8"
val FS2      = "0.9.1"
val FS2Cats  = "0.1.0"

lazy val commonSettings = Seq(
  name    := "Processing"
, version := "0.1.0"
, scalaVersion := "2.11.8"
, libraryDependencies ++= Seq(
    "org.typelevel"  %% "cats"      % "0.7.2"
  , "com.chuusai"    %% "shapeless" % "2.3.2"

  , "org.scalatest"  %% "scalatest" % "3.0.0"  % "test"
  )
, scalacOptions ++= Seq(
      "-deprecation",
      "-encoding", "UTF-8",
      "-feature",
      "-language:existentials",
      "-language:higherKinds",
      "-language:implicitConversions",
      "-language:experimental.macros",
      "-unchecked",
      "-Xfatal-warnings",
      "-Xlint",
      "-Yinline-warnings",
      "-Ywarn-dead-code",
      "-Xfuture")

, libraryDependencies += "commons-io" % "commons-io" % "2.5"

, libraryDependencies += "co.fs2" %% "fs2-core" % FS2
, libraryDependencies += "co.fs2" %% "fs2-cats" % FS2Cats
, libraryDependencies += "co.fs2" %% "fs2-io"   % FS2
)

lazy val root = (project in file("."))
  .settings(commonSettings)
  .settings(
    initialCommands := "import processing._"
  )
