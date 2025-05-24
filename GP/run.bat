@echo off
set WEKA_JAR=weka-stable-3.8.6.jar
set MTJ_JAR=mtj-1.0.1.jar
set CLASS_NAME=GP_Classifier

:: Compile
javac -cp ".;%WEKA_JAR%;%MTJ_JAR%" %CLASS_NAME%.java

:: Run
java -cp ".;%WEKA_JAR%;%MTJ_JAR%" %CLASS_NAME%
