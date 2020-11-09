module mux4to1(out,sel,in1,in2,in3,in4);
input in1,in2,in3,in4;
input[1:0] sel;
output out;
assign out = (in1&(~sel[0])&(~sel[1])) + (in2&sel[0]&(~sel[1])) + (in3&(~sel[0])&sel[1]) + (in4&sel[0]&sel[1]);
endmodule

module bit8_mux4to1(out,sel,in1,in2,in3,in4);
input[7:0] in1,in2,in3,in4;
input[1:0] sel;
output[7:0] out;
generate
	genvar i;
	for(i=0;i<8;i=i+1) begin
		mux4to1 m(out[i],sel,in1[i],in2[i],in3[i],in4[i]);
	end
endgenerate
endmodule

module bit32_mux4to1(out,sel,in1,in2,in3,in4);
input[31:0] in1,in2,in3,in4;
input[1:0] sel;
output[31:0] out;
generate
	genvar i;
	for(i=0;i<4;i=i+1) begin
		bit8_mux4to1 m(out[8*i+7:8*i],sel,in1[8*i+7:8*i],in2[8*i+7:8*i],in3[8*i+7:8*i],in4[8*i+7:8*i]);
	end
endgenerate
endmodule

// module test;
// reg [31:0] inp1,inp2,inp3,inp4;
// reg [1:0] sel;
// wire [31:0] out;
// bit32_mux4to1 M1(out,sel,inp1,inp2,inp3,inp4);
// initial
// begin
// $monitor($time," INP1=%32b INP2=%32b \n\t\t     INP3=%32b INP4=%32b SEL=%2b \n\t\t     OUT=%32b",inp1,inp2,inp3,inp4,sel,out);
// inp1=32'b00001010101010101010101010101010;
// inp2=32'b10010101010101010101010101010101;
// inp3=32'b11001010101010101010101010101010;
// inp4=32'b11100101010101010101010101010101;
// sel=2'b01;
// #50 sel=2'b11;
// #50 sel=2'b00;
// #50 sel=2'b10;
// #1000 $finish;
// end
// endmodule