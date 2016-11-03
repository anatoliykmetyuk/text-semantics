# Char-level identity. Seq[Sentence]; Sentence = Seq[Char]; Char = OH
data = Data(
  # Data load method
  source           = BaseSentencePath
, extractionMethod = line_wise  # list[String]
  
  # Data extraction method
, X                = char_to_one_hot  # Seq[Seq[Char]] = Seq[Seq[Seq[Int]]]
, y                = identity_X       # Seq[Seq[Char]]

, test_data  = fraction(0.15)
, valid_data = fraction(0.15)
)

model_gen  = ModelGen(
  data = data
)
model_lstm = model_gen(lstm)

metrics = [
  accuracy
, Visual(data_type = char_level_one_hot)
]

runner = Runner(
  nb_epoch   = 1000
, batch_size = 64
)

fit_model = runner.fit(data, model_lstm, metrics)

# Char-level prediction
data = data.copy(y = take_x andThen shift(1))
fit_model = runner.fit(data, model_lstm, metrics)

# Char-word-level prediction with encoder-decoder
data = data.copy(X = words andThen char_to_one_hot, y = take_data andThen char_to_one_hot andThen shift(1))
model_lstm = Model(
  layers = [
    lstm_encoder  # Char-level encoder of the words
  ]
)