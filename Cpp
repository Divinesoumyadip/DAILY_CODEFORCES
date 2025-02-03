class Solution {
    // Function to find the height of a binary tree.
    int height(Node node) {
        // code here
        int ans = heightOfTree(node,0);
        return ans-1;
    }
    static int heightOfTree(Node root, int height){
        
        if(root==null) return height;
        
        int temp1=heightOfTree(root.left,height+1);
        int temp2=heightOfTree(root.right,height+1);
        
        height=Math.max(temp1,temp2);
        return height;
    }
}
