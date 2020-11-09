module jk(j,k,clk,q); //won't deal with the pain of reset, I'm sorry
input j,k,clk;
output q;
reg q;
initial
q<=0;
always@(posedge clk)
begin
	case ({j,k}) // note syntax
	2'b00: q <= q;
	2'b01: q <= 0;
	2'b10: q <= 1;
	2'b11: q <= ~q;
	default: q <= 0; // don't forget default!
	endcase // take care to not forget this too
end
endmodule

module ctr(q,clk);
output [3:0]q;
input clk;
jk jk1(1'b1,1'b1,clk,q[0]);
jk jk2(q[0],q[0],clk,q[1]);
jk jk3(q[0]&q[1],q[0]&q[1],clk,q[2]);
jk jk4(q[0]&q[1]&q[2],q[0]&q[1]&q[2],clk,q[3]);
endmodule

module test;
reg clk;
wire[3:0] q;
ctr a(q,clk);
initial
clk=0;
always
#1 clk=~clk;
initial 
$monitor($time," Q=%4b\n",q);
initial
begin
	#31 $finish;
end
endmodule