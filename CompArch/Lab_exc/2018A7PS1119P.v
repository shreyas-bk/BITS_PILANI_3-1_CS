/*
Shreyas Bhat Kera
2018A7PS1119P
*/

module MUX2To1(in0, in1, sel, out);

output out;
input in0, in1, sel;

assign out = sel ? in1 : in0;

endmodule


module MUX16To8 (in1, in2, sel, out) ;

output [7:0] out;
input [7:0] in1, in2; 
input sel;

MUX2To1 m0(in1[0], in2[0], sel, out[0]);
MUX2To1 m1(in1[1], in2[1], sel, out[1]);
MUX2To1 m2(in1[2], in2[2], sel, out[2]);
MUX2To1 m3(in1[3], in2[3], sel, out[3]);
MUX2To1 m4(in1[4], in2[4], sel, out[4]);
MUX2To1 m5(in1[5], in2[5], sel, out[5]);
MUX2To1 m6(in1[6], in2[6], sel, out[6]);
MUX2To1 m7(in1[7], in2[7], sel, out[7]);

endmodule

module RSFF(Q,QBar,S,R,CLK,Reset);

input S,R,CLK,Reset;
output reg Q, QBar;

always@(posedge CLK,posedge Reset)
begin
if(Reset)
begin
Q = 0;
QBar = 1;
end
else if(S == 1)
begin
Q = 1;
QBar = 0;
end
else if(R == 1)
begin
Q = 0;
QBar =1;
end
else if(S == 0 & R == 0) 
begin 
Q <= Q;
QBar <= QBar;
end
end
endmodule

module DFF(
output Q,
output QBar,
input D,
input CLK,
input Reset);

wire DBar;
assign DBar = ~D;
RSFF r0(Q, QBar, D, DBar, CLK, Reset);

endmodule

module Ripple_Counter(
input CLK,
input Reset,
output [3:0] Q);

wire Q0, QN0, Q1, QN1, Q2, QN2, Q3, QN3;

DFF d1(Q0, QN0, QN0, CLK, Reset);
DFF d2(Q1, QN1, QN1, QN0, Reset);
DFF d3(Q2, QN2, QN2, QN1, Reset);
DFF d4(Q3, QN3, QN3, QN2, Reset);

assign Q = {Q3, Q2, Q1, Q0};

endmodule

module Parity_Checker (in, par, out);
output out;
input [7:0]in,par;

assign out = in[0]^in[1]^in[2]^in[3]^in[4]^in[5]^in[6]^in[7];
assign out = (par^out)? 0 : 1;

endmodule

module MEM1(addr,out,parout);
	input [2:0] addr;
	output reg [7:0] out;
	output reg parout;
	reg [7:0] mem [0:7];
	initial begin
		mem[0]=8'b00011111;
		mem[1]=8'b00110001;
		mem[2]=8'b01010011;
		mem[3]=8'b01110101;
		mem[4]=8'b10010111;
		mem[5]=8'b10111001;
		mem[6]=8'b11011011;
		mem[7]=8'b11111101;
		parout=1'b1;
	end
	always@(addr) begin
		case(addr)
			3'b000:out=mem[0];
			3'b001:out=mem[1];			 
			3'b010:out=mem[2];
			3'b011:out=mem[3];
			3'b100:out=mem[4];
			3'b101:out=mem[5];
			3'b110:out=mem[6];
			3'b111:out=mem[7];
		endcase
	end
endmodule

module MEM2(addr,out,parout);
	input [2:0] addr;
	output reg [7:0] out;
	output reg parout;
	reg [7:0] mem [0:7];
	initial begin
		mem[0]=8'b00000000;
		mem[1]=8'b00100010;
		mem[2]=8'b01000100;
		mem[3]=8'b01100110;
		mem[4]=8'b10001000;
		mem[5]=8'b10101010;
		mem[6]=8'b11001100;
		mem[7]=8'b11101110;
		parout=1'b0;
	end
	always@(addr) begin
		case(addr)
			3'b000:out=mem[0];
			3'b001:out=mem[1];			 
			3'b010:out=mem[2];
			3'b011:out=mem[3];
			3'b100:out=mem[4];
			3'b101:out=mem[5];
			3'b110:out=mem[6];
			3'b111:out=mem[7];
		endcase
	end
endmodule


module Fetch_Data(output [7:0] outdata,output parity, input [3:0] select);
	wire [7:0] data1, data2 ;
	wire par1, par2;
	
	MEM1 m1(select[2:0],data1,par1);
	MEM2 m2(select[2:0],data2,par2);
	
	
	MUX16To8 mux1(data1, data2, select[3], outdata); 
	MUX2To1 mux2(par1, par2, select[3], parity);
endmodule


module Design(

	output match,
	input clk,
	input reset
);
	wire [3:0] q;
	
	Ripple_Counter r(clk, reset, q);
	
	wire [7:0] out;
	wire parity;
	
	Fetch_Data f(out, parity, q);
	
	Parity_Checker p(out, parity, match);	
	

endmodule

module TestBench();

	initial begin
		$dumpfile("file.vcd");
		$dumpvars;
	end
	
	
	wire match;
	reg clk;
	reg reset;
	
	Design d(match, clk, reset);
	
	always	
		#0.5 clk = ~clk;
	
	initial begin
			clk = 1'b0;
			reset = 1'b1;
		#1	reset = 1'b0;
		#100 $finish;
	end
	
	initial
		$monitor($time,"  , clk = %b, reset = %b, match = %b", clk, reset, match);
endmodule


