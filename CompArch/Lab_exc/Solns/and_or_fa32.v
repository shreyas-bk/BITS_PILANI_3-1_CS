module bit32AND (out,in1,in2);
input [31:0] in1,in2;
output [31:0] out;
assign {out}=in1 &in2;
endmodule

module bit32OR (out,in1,in2);
input [31:0] in1,in2;
output [31:0] out;
assign {out}=in1|in2;
endmodule

module FA_dataflow (Cout, Sum,In1,In2,Cin);
input[31:0] In1,In2;
input Cin;
output Cout;
output[31:0] Sum;
assign {Cout,Sum}=In1+In2+Cin;
endmodule
