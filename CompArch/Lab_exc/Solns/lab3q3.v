module shiftreg(in, CLK, Q, init, so);
parameter n = 4;
input in;
input CLK;
input[n-1:0] init;
output[n-1:0] Q;
output so;
wire so;
reg[n-1:0] Q;
assign so = Q[0];
always@(init)
Q=init;
always @(posedge CLK)
begin
	Q={in,Q[n-1:1]};
end
endmodule

module dff(d,clk,rst,q);
input clk,d,rst;
output q;
reg q;
always@ (posedge clk or posedge rst)
begin
	if(rst) q <= 1'b0;
	else q <= d;
end
endmodule

module fadder(s,c,x,y,z);
input x,y,z;
output reg s,c;
always @(x or y or z)
begin
s = x^y^z;
c = x&y | y&z | z&x;
end
endmodule

module adder4bit(a,b,sum,CLK,rst);
input[3:0] a;
input[3:0] b;
input CLK,rst;
output[4:0] sum;
reg[4:0] sum;
wire c,q,s,so1,so2;
wire[3:0] q1,q2;
dff dff1(c,CLK,rst,q);
shiftreg s1(s, CLK, q1, a, so1);
shiftreg s2(1'b0, CLK, q2, b, so2);
fadder fadd(s,c,so1,so2,q);
always@(*)
sum = {c,q1};
endmodule

module test;
reg[3:0] a;
reg[3:0] b;
reg clk,rst;
integer i;
wire[4:0] sum;
adder4bit a1(a,b,sum,clk,rst);
initial	
begin
	clk=0;rst=1;
	a=4'b1001;b=4'b1010;
	//$display($time," x=%b y=%b z=%b",a1.s1.so,a1.s2.so,a1.fadd.z);
	for(i=0;i<4;i=i+1) begin
		#1 clk=1;
		//$display($time," x=%b y=%b z=%b",a1.so1,a1.so2,a1.q);
		#1 clk=0;
		rst=0;
		$display($time," a=%4b b=%4b sum=%5b",a,b,sum);
	end
end
endmodule