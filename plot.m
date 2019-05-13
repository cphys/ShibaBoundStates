ClearAll["Global`*"]

(* Parameters for plots *)
mu1 = 0.5;
alph1 = 0.6;
delt1 = .5;
V1 = 0.8;

directoryFunc[muVal_, alphVal_, deltVal_] := 
 directoryFunc[muVal, alphVal, deltVal] = NotebookDirectory[] <> "data/mu" <> ToString[muVal] <> "_alph" <> ToString[alphVal] <> "_delt" <> ToString[deltVal]

potFunc[muVal_, alphVal_, deltVal_, potVal_] := 
 directoryFunc[muVal, alphVal, deltVal] <> "/pot" <> ToString[potVal] <> ".txt"

enFunc[muVal_, alphVal_, deltVal_, potVal_] := 
 directoryFunc[muVal, alphVal, deltVal] <> "/energyValues_pot" <> ToString[potVal] <> ".txt"

potImFunc[muVal_, alphVal_, deltVal_, potVal_] := 
  potImFunc[muVal, alphVal, deltVal, potVal] = ToExpression[Import[potFunc[muVal, alphVal, deltVal, potVal], "Table"]];

enImFunc[muVal_, alphVal_, deltVal_, potVal_] := 
  enImFunc[muVal, alphVal, deltVal, potVal] = Flatten[ToExpression[Import[enFunc[muVal, alphVal, deltVal, potVal], "Table"]]];

AreFunc[muVal_, alphVal_, deltVal_, potVal_] := 
 AreFunc[muVal, alphVal, deltVal, potVal] =
  ArrayFlatten[Table[
    {potImFunc[muVal, alphVal, deltVal, potVal][[j]][[i]],
     enImFunc[muVal, alphVal, deltVal, potVal][[j]]},
    {i, 1, Length[Evaluate[potImFunc[muVal, alphVal, deltVal, potVal]][[1]]]},
    {j, 1, Evaluate[Length[enImFunc[muVal, alphVal, deltVal, potVal]]]}],1]

plotFunction[muVal_, alphVal_, deltVal_, potVal_] :=
 ListLinePlot[{
   SortBy[Table[AreFunc[muVal, alphVal, deltVal, potVal][[i]], {i, 2, Length[AreFunc[muVal, alphVal, deltVal, potVal]]/2}], 1],
   SortBy[Table[AreFunc[muVal, alphVal, deltVal, potVal][[i]], {i, Length[AreFunc[muVal, alphVal, deltVal, potVal]]/2 + 1, Length[AreFunc[muVal, alphVal, deltVal, potVal]]}], 1]},
  PlotRange -> {Automatic, {-0.2125, 0.2125}},
  PlotTheme -> "Scientific",
  FrameLabel -> {"\!\(\*SuperscriptBox[\(g\), \(-1\)]\)", "E"},
  LabelStyle -> {FontFamily -> "Latex", FontSize -> 30},
  ImageSize -> 700,
  PlotStyle -> {{Thickness[0.01], Purple}, {Thickness[0.01], Purple}},
  PlotLegends -> Placed[SwatchLegend[{"V = " <> ToString[potVal], None, None}, LegendMarkerSize -> 20], {.15, .9}]]

exportPlot[muVal_, alphVal_, deltVal_, potVal_] := Export[directoryFunc[muVal, alphVal, deltVal] <> "/plot_V" <> ToString[potVal] <> ".png", plotFunction[muVal, alphVal, deltVal, potVal]]

exportPlot[mu1, alph1, delt1, V1];
