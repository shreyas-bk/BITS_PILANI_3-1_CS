module magcomp4bit(o,a,b);
integer i;
output [2:0]o;
input [3:0]a,b;
reg temp[3:0][2:0]; 
always@(a or b)
for(i=0;i<4;i=i+1)begin
	temp[i][0] = (~a[i])&b[i];
	temp[i][1] = a[i]&(~b[i]);
	temp[i][2] = ~(temp[i][0]|temp[i][1]);
end
assign o[0]=temp[0][2]&temp[1][2]&temp[2][2]&temp[3][2];
assign o[1]=(temp[1][2]&temp[2][2]&temp[3][2]&temp[0][1])|(temp[2][2]&temp[3][2]&temp[1][1])|(temp[3][2]&temp[2][1])|(temp[3][1]);
assign o[2]=(temp[1][2]&temp[2][2]&temp[3][2]&temp[0][0])|(temp[2][2]&temp[3][2]&temp[1][0])|(temp[3][2]&temp[2][0])|(temp[3][0]);
endmodule

module testbench;
reg [3:0]a,b;
wire [2:0]o;
magcomp4bit m(o,a,b);
initial
begin
$monitor($time," Input : a=%4b , b=%4b   Output : %3b",a,b,o);
#0 a=4'b0000;b=4'b0000;
#2 a=4'b0001;b=4'b0010;
#2 a=4'b0011;b=4'b1000;
#2 a=4'b0110;b=4'b0010;
#2 $finish;
end
endmodule